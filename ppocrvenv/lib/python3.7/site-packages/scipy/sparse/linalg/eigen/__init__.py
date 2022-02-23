"""
Sparse Eigenvalue Solvers
-------------------------

The submodules of sparse.linalg.eigen:
    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method

"""
from .arpack import *
from .lobpcg import *

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
