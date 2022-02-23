import numpy as np
from numpy.testing import (assert_allclose,
                           assert_equal)
import scipy.linalg.cython_blas as blas

class TestDGEMM:
    
    def test_transposes(self):

        a = np.arange(12, dtype='d').reshape((3, 4))[:2,:2]
        b = np.arange(1, 13, dtype='d').reshape((4, 3))[:2,:2]
        c = np.empty((2, 4))[:2,:2]

        blas._test_dgemm(1., a, b, 0., c)
        assert_allclose(c, a.dot(b))

        blas._test_dgemm(1., a.T, b, 0., c)
        assert_allclose(c, a.T.dot(b))

        blas._test_dgemm(1., a, b.T, 0., c)
        assert_allclose(c, a.dot(b.T))

        blas._test_dgemm(1., a.T, b.T, 0., c)
        assert_allclose(c, a.T.dot(b.T))

        blas._test_dgemm(1., a, b, 0., c.T)
        assert_allclose(c, a.dot(b).T)

        blas._test_dgemm(1., a.T, b, 0., c.T)
        assert_allclose(c, a.T.dot(b).T)

        blas._test_dgemm(1., a, b.T, 0., c.T)
        assert_allclose(c, a.dot(b.T).T)

        blas._test_dgemm(1., a.T, b.T, 0., c.T)
        assert_allclose(c, a.T.dot(b.T).T)
    
    def test_shapes(self):
        a = np.arange(6, dtype='d').reshape((3, 2))
        b = np.arange(-6, 2, dtype='d').reshape((2, 4))
        c = np.empty((3, 4))

        blas._test_dgemm(1., a, b, 0., c)
        assert_allclose(c, a.dot(b))

        blas._test_dgemm(1., b.T, a.T, 0., c.T)
        assert_allclose(c, b.T.dot(a.T).T)
        
class TestWfuncPointers:
    """ Test the function pointers that are expected to fail on
    Mac OS X without the additional entry statement in their definitions
    in fblas_l1.pyf.src. """

    def test_complex_args(self):

        cx = np.array([.5 + 1.j, .25 - .375j, 12.5 - 4.j], np.complex64)
        cy = np.array([.8 + 2.j, .875 - .625j, -1. + 2.j], np.complex64)

        assert_allclose(blas._test_cdotc(cx, cy),
                        -17.6468753815+21.3718757629j, 5)
        assert_allclose(blas._test_cdotu(cx, cy),
                        -6.11562538147+30.3156242371j, 5)

        assert_equal(blas._test_icamax(cx), 3)

        assert_allclose(blas._test_scasum(cx), 18.625, 5)
        assert_allclose(blas._test_scnrm2(cx), 13.1796483994, 5)

        assert_allclose(blas._test_cdotc(cx[::2], cy[::2]),
                        -18.1000003815+21.2000007629j, 5)
        assert_allclose(blas._test_cdotu(cx[::2], cy[::2]),
                        -6.10000038147+30.7999992371j, 5)
        assert_allclose(blas._test_scasum(cx[::2]), 18., 5)
        assert_allclose(blas._test_scnrm2(cx[::2]), 13.1719398499, 5)
    
    def test_double_args(self):

        x = np.array([5., -3, -.5], np.float64)
        y = np.array([2, 1, .5], np.float64)

        assert_allclose(blas._test_dasum(x), 8.5, 10)
        assert_allclose(blas._test_ddot(x, y), 6.75, 10)
        assert_allclose(blas._test_dnrm2(x), 5.85234975815, 10)

        assert_allclose(blas._test_dasum(x[::2]), 5.5, 10)
        assert_allclose(blas._test_ddot(x[::2], y[::2]), 9.75, 10)
        assert_allclose(blas._test_dnrm2(x[::2]), 5.0249376297, 10)

        assert_equal(blas._test_idamax(x), 1)

    def test_float_args(self):

        x = np.array([5., -3, -.5], np.float32)
        y = np.array([2, 1, .5], np.float32)

        assert_equal(blas._test_isamax(x), 1)

        assert_allclose(blas._test_sasum(x), 8.5, 5)
        assert_allclose(blas._test_sdot(x, y), 6.75, 5)
        assert_allclose(blas._test_snrm2(x), 5.85234975815, 5)

        assert_allclose(blas._test_sasum(x[::2]), 5.5, 5)
        assert_allclose(blas._test_sdot(x[::2], y[::2]), 9.75, 5)
        assert_allclose(blas._test_snrm2(x[::2]), 5.0249376297, 5)

    def test_double_complex_args(self):

        cx = np.array([.5 + 1.j, .25 - .375j, 13. - 4.j], np.complex128)
        cy = np.array([.875 + 2.j, .875 - .625j, -1. + 2.j], np.complex128)

        assert_equal(blas._test_izamax(cx), 3)

        assert_allclose(blas._test_zdotc(cx, cy), -18.109375+22.296875j, 10)
        assert_allclose(blas._test_zdotu(cx, cy), -6.578125+31.390625j, 10)

        assert_allclose(blas._test_zdotc(cx[::2], cy[::2]),
                        -18.5625+22.125j, 10)
        assert_allclose(blas._test_zdotu(cx[::2], cy[::2]),
                        -6.5625+31.875j, 10)

