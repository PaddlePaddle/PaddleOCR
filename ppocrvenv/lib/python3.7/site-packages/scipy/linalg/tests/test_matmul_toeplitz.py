"""Test functions for linalg.matmul_toeplitz function
"""

import numpy as np
from scipy.linalg import toeplitz, matmul_toeplitz

from pytest import raises as assert_raises
from numpy.testing import assert_allclose


class TestMatmulToeplitz:

    def setup_method(self):
        self.rng = np.random.RandomState(42)
        self.tolerance = 1.5e-13

    def test_real(self):
        cases = []

        n = 1
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n)
        x = self.rng.normal(size=(n, 1))
        cases.append((x, c, r, False))

        n = 2
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n)
        x = self.rng.normal(size=(n, 1))
        cases.append((x, c, r, False))

        n = 101
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n)
        x = self.rng.normal(size=(n, 1))
        cases.append((x, c, r, True))

        n = 1000
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n)
        x = self.rng.normal(size=(n, 1))
        cases.append((x, c, r, False))

        n = 100
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n)
        x = self.rng.normal(size=(n, self.rng.randint(1, 10)))
        cases.append((x, c, r, False))

        n = 100
        c = self.rng.normal(size=(n, 1))
        r = self.rng.normal(size=(n, 1))
        x = self.rng.normal(size=(n, self.rng.randint(1, 10)))
        cases.append((x, c, r, True))

        n = 100
        c = self.rng.normal(size=(n, 1))
        r = None
        x = self.rng.normal(size=(n, self.rng.randint(1, 10)))
        cases.append((x, c, r, True, -1))

        n = 100
        c = self.rng.normal(size=(n, 1))
        r = None
        x = self.rng.normal(size=n)
        cases.append((x, c, r, False))

        n = 101
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n-27)
        x = self.rng.normal(size=(n-27, 1))
        cases.append((x, c, r, True))

        n = 100
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n//4)
        x = self.rng.normal(size=(n//4, self.rng.randint(1, 10)))
        cases.append((x, c, r, True))

        [self.do(*i) for i in cases]

    def test_complex(self):
        n = 127
        c = self.rng.normal(size=(n, 1)) + self.rng.normal(size=(n, 1))*1j
        r = self.rng.normal(size=(n, 1)) + self.rng.normal(size=(n, 1))*1j
        x = self.rng.normal(size=(n, 3)) + self.rng.normal(size=(n, 3))*1j
        self.do(x, c, r, False)

        n = 100
        c = self.rng.normal(size=(n, 1)) + self.rng.normal(size=(n, 1))*1j
        r = self.rng.normal(size=(n//2, 1)) +\
            self.rng.normal(size=(n//2, 1))*1j
        x = self.rng.normal(size=(n//2, 3)) +\
            self.rng.normal(size=(n//2, 3))*1j
        self.do(x, c, r, False)

    def test_exceptions(self):

        n = 100
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=2*n)
        x = self.rng.normal(size=n)
        assert_raises(ValueError, matmul_toeplitz, (c, r), x, True)

        n = 100
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n)
        x = self.rng.normal(size=n-1)
        assert_raises(ValueError, matmul_toeplitz, (c, r), x, True)

        n = 100
        c = self.rng.normal(size=n)
        r = self.rng.normal(size=n//2)
        x = self.rng.normal(size=n//2-1)
        assert_raises(ValueError, matmul_toeplitz, (c, r), x, True)

    # For toeplitz matrices, matmul_toeplitz() should be equivalent to @.
    def do(self, x, c, r=None, check_finite=False, workers=None):
        if r is None:
            actual = matmul_toeplitz(c, x, check_finite, workers)
        else:
            actual = matmul_toeplitz((c, r), x, check_finite)
        desired = toeplitz(c, r) @ x
        assert_allclose(actual, desired,
            rtol=self.tolerance, atol=self.tolerance)
