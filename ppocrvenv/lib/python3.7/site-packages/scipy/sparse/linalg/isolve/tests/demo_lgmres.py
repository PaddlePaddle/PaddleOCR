import scipy.sparse.linalg as la
import scipy.io as io
import numpy as np
import sys

#problem = "SPARSKIT/drivcav/e05r0100"
problem = "SPARSKIT/drivcav/e05r0200"
#problem = "Harwell-Boeing/sherman/sherman1"
#problem = "misc/hamm/add32"

mm = np.lib._datasource.Repository('ftp://math.nist.gov/pub/MatrixMarket2/')
f = mm.open('%s.mtx.gz' % problem)
Am = io.mmread(f).tocsr()
f.close()

f = mm.open('%s_rhs1.mtx.gz' % problem)
b = np.array(io.mmread(f)).ravel()
f.close()

count = [0]


def matvec(v):
    count[0] += 1
    sys.stderr.write('%d\r' % count[0])
    return Am*v


A = la.LinearOperator(matvec=matvec, shape=Am.shape, dtype=Am.dtype)

M = 100

print("MatrixMarket problem %s" % problem)
print("Invert %d x %d matrix; nnz = %d" % (Am.shape[0], Am.shape[1], Am.nnz))

count[0] = 0
x0, info = la.gmres(A, b, restrt=M, tol=1e-14)
count_0 = count[0]
err0 = np.linalg.norm(Am*x0 - b) / np.linalg.norm(b)
print("GMRES(%d):" % M, count_0, "matvecs, residual", err0)
if info != 0:
    print("Didn't converge")

count[0] = 0
x1, info = la.lgmres(A, b, inner_m=M-6*2, outer_k=6, tol=1e-14)
count_1 = count[0]
err1 = np.linalg.norm(Am*x1 - b) / np.linalg.norm(b)
print("LGMRES(%d,6) [same memory req.]:" % (M-2*6), count_1,
      "matvecs, residual:", err1)
if info != 0:
    print("Didn't converge")

count[0] = 0
x2, info = la.lgmres(A, b, inner_m=M-6, outer_k=6, tol=1e-14)
count_2 = count[0]
err2 = np.linalg.norm(Am*x2 - b) / np.linalg.norm(b)
print("LGMRES(%d,6) [same subspace size]:" % (M-6), count_2,
      "matvecs, residual:", err2)
if info != 0:
    print("Didn't converge")
