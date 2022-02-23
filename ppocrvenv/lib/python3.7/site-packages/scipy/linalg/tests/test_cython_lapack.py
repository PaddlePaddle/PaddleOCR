from numpy.testing import assert_allclose
from scipy.linalg import cython_lapack as cython_lapack
from scipy.linalg import lapack


class TestLamch:

    def test_slamch(self):
        for c in [b'e', b's', b'b', b'p', b'n', b'r', b'm', b'u', b'l', b'o']:
            assert_allclose(cython_lapack._test_slamch(c),
                            lapack.slamch(c))

    def test_dlamch(self):
        for c in [b'e', b's', b'b', b'p', b'n', b'r', b'm', b'u', b'l', b'o']:
            assert_allclose(cython_lapack._test_dlamch(c),
                            lapack.dlamch(c))

