"""
Linear Solvers
==============

The default solver is SuperLU (included in the scipy distribution),
which can solve real or complex linear systems in both single and
double precisions.  It is automatically replaced by UMFPACK, if
available.  Note that UMFPACK works in double precision only, so
switch it off by::

    >>> use_solver(useUmfpack=False)

to solve in the single precision. See also use_solver documentation.

Example session::

    >>> from scipy.sparse import csc_matrix, spdiags
    >>> from numpy import array
    >>> from scipy.sparse.linalg import spsolve, use_solver
    >>>
    >>> print("Inverting a sparse linear system:")
    >>> print("The sparse matrix (constructed from diagonals):")
    >>> a = spdiags([[1, 2, 3, 4, 5], [6, 5, 8, 9, 10]], [0, 1], 5, 5)
    >>> b = array([1, 2, 3, 4, 5])
    >>> print("Solve: single precision complex:")
    >>> use_solver( useUmfpack = False )
    >>> a = a.astype('F')
    >>> x = spsolve(a, b)
    >>> print(x)
    >>> print("Error: ", a*x-b)
    >>>
    >>> print("Solve: double precision complex:")
    >>> use_solver( useUmfpack = True )
    >>> a = a.astype('D')
    >>> x = spsolve(a, b)
    >>> print(x)
    >>> print("Error: ", a*x-b)
    >>>
    >>> print("Solve: double precision:")
    >>> a = a.astype('d')
    >>> x = spsolve(a, b)
    >>> print(x)
    >>> print("Error: ", a*x-b)
    >>>
    >>> print("Solve: single precision:")
    >>> use_solver( useUmfpack = False )
    >>> a = a.astype('f')
    >>> x = spsolve(a, b.astype('f'))
    >>> print(x)
    >>> print("Error: ", a*x-b)

"""

#import umfpack
#__doc__ = '\n\n'.join( (__doc__,  umfpack.__doc__) )
#del umfpack

from .linsolve import *
from ._superlu import SuperLU
from . import _add_newdocs

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
