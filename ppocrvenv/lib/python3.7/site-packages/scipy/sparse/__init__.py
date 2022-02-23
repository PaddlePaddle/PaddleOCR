"""
=====================================
Sparse matrices (:mod:`scipy.sparse`)
=====================================

.. currentmodule:: scipy.sparse

SciPy 2-D sparse matrix package for numeric data.

Contents
========

Sparse matrix classes
---------------------

.. autosummary::
   :toctree: generated/

   bsr_matrix - Block Sparse Row matrix
   coo_matrix - A sparse matrix in COOrdinate format
   csc_matrix - Compressed Sparse Column matrix
   csr_matrix - Compressed Sparse Row matrix
   dia_matrix - Sparse matrix with DIAgonal storage
   dok_matrix - Dictionary Of Keys based sparse matrix
   lil_matrix - Row-based list of lists sparse matrix
   spmatrix - Sparse matrix base class

Functions
---------

Building sparse matrices:

.. autosummary::
   :toctree: generated/

   eye - Sparse MxN matrix whose k-th diagonal is all ones
   identity - Identity matrix in sparse format
   kron - kronecker product of two sparse matrices
   kronsum - kronecker sum of sparse matrices
   diags - Return a sparse matrix from diagonals
   spdiags - Return a sparse matrix from diagonals
   block_diag - Build a block diagonal sparse matrix
   tril - Lower triangular portion of a matrix in sparse format
   triu - Upper triangular portion of a matrix in sparse format
   bmat - Build a sparse matrix from sparse sub-blocks
   hstack - Stack sparse matrices horizontally (column wise)
   vstack - Stack sparse matrices vertically (row wise)
   rand - Random values in a given shape
   random - Random values in a given shape

Save and load sparse matrices:

.. autosummary::
   :toctree: generated/

   save_npz - Save a sparse matrix to a file using ``.npz`` format.
   load_npz - Load a sparse matrix from a file using ``.npz`` format.

Sparse matrix tools:

.. autosummary::
   :toctree: generated/

   find

Identifying sparse matrices:

.. autosummary::
   :toctree: generated/

   issparse
   isspmatrix
   isspmatrix_csc
   isspmatrix_csr
   isspmatrix_bsr
   isspmatrix_lil
   isspmatrix_dok
   isspmatrix_coo
   isspmatrix_dia

Submodules
----------

.. autosummary::

   csgraph - Compressed sparse graph routines
   linalg - sparse linear algebra routines

Exceptions
----------

.. autosummary::
   :toctree: generated/

   SparseEfficiencyWarning
   SparseWarning


Usage information
=================

There are seven available sparse matrix types:

    1. csc_matrix: Compressed Sparse Column format
    2. csr_matrix: Compressed Sparse Row format
    3. bsr_matrix: Block Sparse Row format
    4. lil_matrix: List of Lists format
    5. dok_matrix: Dictionary of Keys format
    6. coo_matrix: COOrdinate format (aka IJV, triplet format)
    7. dia_matrix: DIAgonal format

To construct a matrix efficiently, use either dok_matrix or lil_matrix.
The lil_matrix class supports basic slicing and fancy indexing with a
similar syntax to NumPy arrays. As illustrated below, the COO format
may also be used to efficiently construct matrices. Despite their
similarity to NumPy arrays, it is **strongly discouraged** to use NumPy
functions directly on these matrices because NumPy may not properly convert
them for computations, leading to unexpected (and incorrect) results. If you
do want to apply a NumPy function to these matrices, first check if SciPy has
its own implementation for the given sparse matrix class, or **convert the
sparse matrix to a NumPy array** (e.g., using the `toarray()` method of the
class) first before applying the method.

To perform manipulations such as multiplication or inversion, first
convert the matrix to either CSC or CSR format. The lil_matrix format is
row-based, so conversion to CSR is efficient, whereas conversion to CSC
is less so.

All conversions among the CSR, CSC, and COO formats are efficient,
linear-time operations.

Matrix vector product
---------------------
To do a vector product between a sparse matrix and a vector simply use
the matrix `dot` method, as described in its docstring:

>>> import numpy as np
>>> from scipy.sparse import csr_matrix
>>> A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
>>> v = np.array([1, 0, -1])
>>> A.dot(v)
array([ 1, -3, -1], dtype=int64)

.. warning:: As of NumPy 1.7, `np.dot` is not aware of sparse matrices,
  therefore using it will result on unexpected results or errors.
  The corresponding dense array should be obtained first instead:

  >>> np.dot(A.toarray(), v)
  array([ 1, -3, -1], dtype=int64)

  but then all the performance advantages would be lost.

The CSR format is specially suitable for fast matrix vector products.

Example 1
---------
Construct a 1000x1000 lil_matrix and add some values to it:

>>> from scipy.sparse import lil_matrix
>>> from scipy.sparse.linalg import spsolve
>>> from numpy.linalg import solve, norm
>>> from numpy.random import rand

>>> A = lil_matrix((1000, 1000))
>>> A[0, :100] = rand(100)
>>> A[1, 100:200] = A[0, :100]
>>> A.setdiag(rand(1000))

Now convert it to CSR format and solve A x = b for x:

>>> A = A.tocsr()
>>> b = rand(1000)
>>> x = spsolve(A, b)

Convert it to a dense matrix and solve, and check that the result
is the same:

>>> x_ = solve(A.toarray(), b)

Now we can compute norm of the error with:

>>> err = norm(x-x_)
>>> err < 1e-10
True

It should be small :)


Example 2
---------

Construct a matrix in COO format:

>>> from scipy import sparse
>>> from numpy import array
>>> I = array([0,3,1,0])
>>> J = array([0,3,1,2])
>>> V = array([4,5,7,9])
>>> A = sparse.coo_matrix((V,(I,J)),shape=(4,4))

Notice that the indices do not need to be sorted.

Duplicate (i,j) entries are summed when converting to CSR or CSC.

>>> I = array([0,0,1,3,1,0,0])
>>> J = array([0,2,1,3,1,0,0])
>>> V = array([1,1,1,1,1,1,1])
>>> B = sparse.coo_matrix((V,(I,J)),shape=(4,4)).tocsr()

This is useful for constructing finite-element stiffness and mass matrices.

Further details
---------------

CSR column indices are not necessarily sorted. Likewise for CSC row
indices. Use the .sorted_indices() and .sort_indices() methods when
sorted indices are required (e.g., when passing data to other libraries).

"""

# Original code by Travis Oliphant.
# Modified and extended by Ed Schofield, Robert Cimrman,
# Nathan Bell, and Jake Vanderplas.

import warnings as _warnings

from .base import *
from .csr import *
from .csc import *
from .lil import *
from .dok import *
from .coo import *
from .dia import *
from .bsr import *
from .construct import *
from .extract import *
from ._matrix_io import *

# For backward compatibility with v0.19.
from . import csgraph

__all__ = [s for s in dir() if not s.startswith('_')]

# Filter PendingDeprecationWarning for np.matrix introduced with numpy 1.15
_warnings.filterwarnings('ignore', message='the matrix subclass is not the recommended way')

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
