"""
Sparse linear algebra (:mod:`scipy.sparse.linalg`)
==================================================

.. currentmodule:: scipy.sparse.linalg

Abstract linear operators
-------------------------

.. autosummary::
   :toctree: generated/

   LinearOperator -- abstract representation of a linear operator
   aslinearoperator -- convert an object to an abstract linear operator

Matrix Operations
-----------------

.. autosummary::
   :toctree: generated/

   inv -- compute the sparse matrix inverse
   expm -- compute the sparse matrix exponential
   expm_multiply -- compute the product of a matrix exponential and a matrix

Matrix norms
------------

.. autosummary::
   :toctree: generated/

   norm -- Norm of a sparse matrix
   onenormest -- Estimate the 1-norm of a sparse matrix

Solving linear problems
-----------------------

Direct methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   spsolve -- Solve the sparse linear system Ax=b
   spsolve_triangular -- Solve the sparse linear system Ax=b for a triangular matrix
   factorized -- Pre-factorize matrix to a function solving a linear system
   MatrixRankWarning -- Warning on exactly singular matrices
   use_solver -- Select direct solver to use

Iterative methods for linear equation systems:

.. autosummary::
   :toctree: generated/

   bicg -- Use BIConjugate Gradient iteration to solve A x = b
   bicgstab -- Use BIConjugate Gradient STABilized iteration to solve A x = b
   cg -- Use Conjugate Gradient iteration to solve A x = b
   cgs -- Use Conjugate Gradient Squared iteration to solve A x = b
   gmres -- Use Generalized Minimal RESidual iteration to solve A x = b
   lgmres -- Solve a matrix equation using the LGMRES algorithm
   minres -- Use MINimum RESidual iteration to solve Ax = b
   qmr -- Use Quasi-Minimal Residual iteration to solve A x = b
   gcrotmk -- Solve a matrix equation using the GCROT(m,k) algorithm

Iterative methods for least-squares problems:

.. autosummary::
   :toctree: generated/

   lsqr -- Find the least-squares solution to a sparse linear equation system
   lsmr -- Find the least-squares solution to a sparse linear equation system

Matrix factorizations
---------------------

Eigenvalue problems:

.. autosummary::
   :toctree: generated/

   eigs -- Find k eigenvalues and eigenvectors of the square matrix A
   eigsh -- Find k eigenvalues and eigenvectors of a symmetric matrix
   lobpcg -- Solve symmetric partial eigenproblems with optional preconditioning

Singular values problems:

.. autosummary::
   :toctree: generated/

   svds -- Compute k singular values/vectors for a sparse matrix

Complete or incomplete LU factorizations

.. autosummary::
   :toctree: generated/

   splu -- Compute a LU decomposition for a sparse matrix
   spilu -- Compute an incomplete LU decomposition for a sparse matrix
   SuperLU -- Object representing an LU factorization

Exceptions
----------

.. autosummary::
   :toctree: generated/

   ArpackNoConvergence
   ArpackError

"""

from .isolve import *
from .dsolve import *
from .interface import *
from .eigen import *
from .matfuncs import *
from ._onenormest import *
from ._norm import *
from ._expm_multiply import *

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
