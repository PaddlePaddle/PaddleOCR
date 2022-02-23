"""
====================================
Linear algebra (:mod:`scipy.linalg`)
====================================

.. currentmodule:: scipy.linalg

Linear algebra functions.

.. eventually, we should replace the numpy.linalg HTML link with just `numpy.linalg`

.. seealso::

   `numpy.linalg <https://www.numpy.org/devdocs/reference/routines.linalg.html>`__
   for more linear algebra functions. Note that
   although `scipy.linalg` imports most of them, identically named
   functions from `scipy.linalg` may offer more or slightly differing
   functionality.


Basics
======

.. autosummary::
   :toctree: generated/

   inv - Find the inverse of a square matrix
   solve - Solve a linear system of equations
   solve_banded - Solve a banded linear system
   solveh_banded - Solve a Hermitian or symmetric banded system
   solve_circulant - Solve a circulant system
   solve_triangular - Solve a triangular matrix
   solve_toeplitz - Solve a toeplitz matrix
   matmul_toeplitz - Multiply a Toeplitz matrix with an array.
   det - Find the determinant of a square matrix
   norm - Matrix and vector norm
   lstsq - Solve a linear least-squares problem
   pinv - Pseudo-inverse (Moore-Penrose) using lstsq
   pinv2 - Pseudo-inverse using svd
   pinvh - Pseudo-inverse of hermitian matrix
   kron - Kronecker product of two arrays
   khatri_rao - Khatri-Rao product of two arrays
   tril - Construct a lower-triangular matrix from a given matrix
   triu - Construct an upper-triangular matrix from a given matrix
   orthogonal_procrustes - Solve an orthogonal Procrustes problem
   matrix_balance - Balance matrix entries with a similarity transformation
   subspace_angles - Compute the subspace angles between two matrices
   LinAlgError
   LinAlgWarning

Eigenvalue Problems
===================

.. autosummary::
   :toctree: generated/

   eig - Find the eigenvalues and eigenvectors of a square matrix
   eigvals - Find just the eigenvalues of a square matrix
   eigh - Find the e-vals and e-vectors of a Hermitian or symmetric matrix
   eigvalsh - Find just the eigenvalues of a Hermitian or symmetric matrix
   eig_banded - Find the eigenvalues and eigenvectors of a banded matrix
   eigvals_banded - Find just the eigenvalues of a banded matrix
   eigh_tridiagonal - Find the eigenvalues and eigenvectors of a tridiagonal matrix
   eigvalsh_tridiagonal - Find just the eigenvalues of a tridiagonal matrix

Decompositions
==============

.. autosummary::
   :toctree: generated/

   lu - LU decomposition of a matrix
   lu_factor - LU decomposition returning unordered matrix and pivots
   lu_solve - Solve Ax=b using back substitution with output of lu_factor
   svd - Singular value decomposition of a matrix
   svdvals - Singular values of a matrix
   diagsvd - Construct matrix of singular values from output of svd
   orth - Construct orthonormal basis for the range of A using svd
   null_space - Construct orthonormal basis for the null space of A using svd
   ldl - LDL.T decomposition of a Hermitian or a symmetric matrix.
   cholesky - Cholesky decomposition of a matrix
   cholesky_banded - Cholesky decomp. of a sym. or Hermitian banded matrix
   cho_factor - Cholesky decomposition for use in solving a linear system
   cho_solve - Solve previously factored linear system
   cho_solve_banded - Solve previously factored banded linear system
   polar - Compute the polar decomposition.
   qr - QR decomposition of a matrix
   qr_multiply - QR decomposition and multiplication by Q
   qr_update - Rank k QR update
   qr_delete - QR downdate on row or column deletion
   qr_insert - QR update on row or column insertion
   rq - RQ decomposition of a matrix
   qz - QZ decomposition of a pair of matrices
   ordqz - QZ decomposition of a pair of matrices with reordering
   schur - Schur decomposition of a matrix
   rsf2csf - Real to complex Schur form
   hessenberg - Hessenberg form of a matrix
   cdf2rdf - Complex diagonal form to real diagonal block form
   cossin - Cosine sine decomposition of a unitary or orthogonal matrix

.. seealso::

   `scipy.linalg.interpolative` -- Interpolative matrix decompositions


Matrix Functions
================

.. autosummary::
   :toctree: generated/

   expm - Matrix exponential
   logm - Matrix logarithm
   cosm - Matrix cosine
   sinm - Matrix sine
   tanm - Matrix tangent
   coshm - Matrix hyperbolic cosine
   sinhm - Matrix hyperbolic sine
   tanhm - Matrix hyperbolic tangent
   signm - Matrix sign
   sqrtm - Matrix square root
   funm - Evaluating an arbitrary matrix function
   expm_frechet - Frechet derivative of the matrix exponential
   expm_cond - Relative condition number of expm in the Frobenius norm
   fractional_matrix_power - Fractional matrix power


Matrix Equation Solvers
=======================

.. autosummary::
   :toctree: generated/

   solve_sylvester - Solve the Sylvester matrix equation
   solve_continuous_are - Solve the continuous-time algebraic Riccati equation
   solve_discrete_are - Solve the discrete-time algebraic Riccati equation
   solve_continuous_lyapunov - Solve the continuous-time Lyapunov equation
   solve_discrete_lyapunov - Solve the discrete-time Lyapunov equation


Sketches and Random Projections
===============================

.. autosummary::
   :toctree: generated/

   clarkson_woodruff_transform - Applies the Clarkson Woodruff Sketch (a.k.a CountMin Sketch)

Special Matrices
================

.. autosummary::
   :toctree: generated/

   block_diag - Construct a block diagonal matrix from submatrices
   circulant - Circulant matrix
   companion - Companion matrix
   convolution_matrix - Convolution matrix
   dft - Discrete Fourier transform matrix
   fiedler - Fiedler matrix
   fiedler_companion - Fiedler companion matrix
   hadamard - Hadamard matrix of order 2**n
   hankel - Hankel matrix
   helmert - Helmert matrix
   hilbert - Hilbert matrix
   invhilbert - Inverse Hilbert matrix
   leslie - Leslie matrix
   pascal - Pascal matrix
   invpascal - Inverse Pascal matrix
   toeplitz - Toeplitz matrix
   tri - Construct a matrix filled with ones at and below a given diagonal

Low-level routines
==================

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   get_lapack_funcs
   find_best_blas_type

.. seealso::

   `scipy.linalg.blas` -- Low-level BLAS functions

   `scipy.linalg.lapack` -- Low-level LAPACK functions

   `scipy.linalg.cython_blas` -- Low-level BLAS functions for Cython

   `scipy.linalg.cython_lapack` -- Low-level LAPACK functions for Cython

"""  # noqa: E501

from .misc import *
from .basic import *
from .decomp import *
from .decomp_lu import *
from ._decomp_ldl import *
from .decomp_cholesky import *
from .decomp_qr import *
from ._decomp_qz import *
from .decomp_svd import *
from .decomp_schur import *
from ._decomp_polar import *
from .matfuncs import *
from .blas import *
from .lapack import *
from .special_matrices import *
from ._solvers import *
from ._procrustes import *
from ._decomp_update import *
from ._sketches import *
from ._decomp_cossin import *

__all__ = [s for s in dir() if not s.startswith('_')]


from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
