"""
Eigenvalue solver using iterative methods.

Find k eigenvectors and eigenvalues of a matrix A using the
Arnoldi/Lanczos iterative methods from ARPACK [1]_,[2]_.

These methods are most useful for large sparse matrices.

  - eigs(A,k)
  - eigsh(A,k)

References
----------
.. [1] ARPACK Software, http://www.caam.rice.edu/software/ARPACK/
.. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
   Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
   Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

"""
from .arpack import *
