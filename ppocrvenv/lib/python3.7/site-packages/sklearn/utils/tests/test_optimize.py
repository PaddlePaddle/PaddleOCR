import numpy as np

from sklearn.utils.optimize import _newton_cg
from scipy.optimize import fmin_ncg

from sklearn.utils._testing import assert_array_almost_equal


def test_newton_cg():
    # Test that newton_cg gives same result as scipy's fmin_ncg

    rng = np.random.RandomState(0)
    A = rng.normal(size=(10, 10))
    x0 = np.ones(10)

    def func(x):
        Ax = A.dot(x)
        return 0.5 * (Ax).dot(Ax)

    def grad(x):
        return A.T.dot(A.dot(x))

    def hess(x, p):
        return p.dot(A.T.dot(A.dot(x.all())))

    def grad_hess(x):
        return grad(x), lambda x: A.T.dot(A.dot(x))

    assert_array_almost_equal(
        _newton_cg(grad_hess, func, grad, x0, tol=1e-10)[0],
        fmin_ncg(f=func, x0=x0, fprime=grad, fhess_p=hess),
    )
