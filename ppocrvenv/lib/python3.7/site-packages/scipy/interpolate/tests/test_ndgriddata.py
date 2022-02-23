import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_allclose
from pytest import raises as assert_raises

from scipy.interpolate import griddata, NearestNDInterpolator


class TestGriddata:
    def test_fill_value(self):
        x = [(0,0), (0,1), (1,0)]
        y = [1, 2, 3]

        yi = griddata(x, y, [(1,1), (1,2), (0,0)], fill_value=-1)
        assert_array_equal(yi, [-1., -1, 1])

        yi = griddata(x, y, [(1,1), (1,2), (0,0)])
        assert_array_equal(yi, [np.nan, np.nan, 1])

    def test_alternative_call(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = (np.arange(x.shape[0], dtype=np.double)[:,None]
             + np.array([0,1])[None,:])

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata((x[:,0], x[:,1]), y, (x[:,0], x[:,1]), method=method,
                              rescale=rescale)
                assert_allclose(y, yi, atol=1e-14, err_msg=msg)

    def test_multivalue_2d(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = (np.arange(x.shape[0], dtype=np.double)[:,None]
             + np.array([0,1])[None,:])

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata(x, y, x, method=method, rescale=rescale)
                assert_allclose(y, yi, atol=1e-14, err_msg=msg)

    def test_multipoint_2d(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = np.arange(x.shape[0], dtype=np.double)

        xi = x[:,None,:] + np.array([0,0,0])[None,:,None]

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata(x, y, xi, method=method, rescale=rescale)

                assert_equal(yi.shape, (5, 3), err_msg=msg)
                assert_allclose(yi, np.tile(y[:,None], (1, 3)),
                                atol=1e-14, err_msg=msg)

    def test_complex_2d(self):
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = np.arange(x.shape[0], dtype=np.double)
        y = y - 2j*y[::-1]

        xi = x[:,None,:] + np.array([0,0,0])[None,:,None]

        for method in ('nearest', 'linear', 'cubic'):
            for rescale in (True, False):
                msg = repr((method, rescale))
                yi = griddata(x, y, xi, method=method, rescale=rescale)

                assert_equal(yi.shape, (5, 3), err_msg=msg)
                assert_allclose(yi, np.tile(y[:,None], (1, 3)),
                                atol=1e-14, err_msg=msg)

    def test_1d(self):
        x = np.array([1, 2.5, 3, 4.5, 5, 6])
        y = np.array([1, 2, 0, 3.9, 2, 1])

        for method in ('nearest', 'linear', 'cubic'):
            assert_allclose(griddata(x, y, x, method=method), y,
                            err_msg=method, atol=1e-14)
            assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
                            err_msg=method, atol=1e-14)
            assert_allclose(griddata((x,), y, (x,), method=method), y,
                            err_msg=method, atol=1e-14)

    def test_1d_borders(self):
        # Test for nearest neighbor case with xi outside
        # the range of the values.
        x = np.array([1, 2.5, 3, 4.5, 5, 6])
        y = np.array([1, 2, 0, 3.9, 2, 1])
        xi = np.array([0.9, 6.5])
        yi_should = np.array([1.0, 1.0])

        method = 'nearest'
        assert_allclose(griddata(x, y, xi,
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)
        assert_allclose(griddata(x.reshape(6, 1), y, xi,
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)
        assert_allclose(griddata((x, ), y, (xi, ),
                                 method=method), yi_should,
                        err_msg=method,
                        atol=1e-14)

    def test_1d_unsorted(self):
        x = np.array([2.5, 1, 4.5, 5, 6, 3])
        y = np.array([1, 2, 0, 3.9, 2, 1])

        for method in ('nearest', 'linear', 'cubic'):
            assert_allclose(griddata(x, y, x, method=method), y,
                            err_msg=method, atol=1e-10)
            assert_allclose(griddata(x.reshape(6, 1), y, x, method=method), y,
                            err_msg=method, atol=1e-10)
            assert_allclose(griddata((x,), y, (x,), method=method), y,
                            err_msg=method, atol=1e-10)

    def test_square_rescale_manual(self):
        points = np.array([(0,0), (0,100), (10,100), (10,0), (1, 5)], dtype=np.double)
        points_rescaled = np.array([(0,0), (0,1), (1,1), (1,0), (0.1, 0.05)], dtype=np.double)
        values = np.array([1., 2., -3., 5., 9.], dtype=np.double)

        xx, yy = np.broadcast_arrays(np.linspace(0, 10, 14)[:,None],
                                     np.linspace(0, 100, 14)[None,:])
        xx = xx.ravel()
        yy = yy.ravel()
        xi = np.array([xx, yy]).T.copy()

        for method in ('nearest', 'linear', 'cubic'):
            msg = method
            zi = griddata(points_rescaled, values, xi/np.array([10, 100.]),
                          method=method)
            zi_rescaled = griddata(points, values, xi, method=method,
                                   rescale=True)
            assert_allclose(zi, zi_rescaled, err_msg=msg,
                            atol=1e-12)

    def test_xi_1d(self):
        # Check that 1-D xi is interpreted as a coordinate
        x = np.array([(0,0), (-0.5,-0.5), (-0.5,0.5), (0.5, 0.5), (0.25, 0.3)],
                     dtype=np.double)
        y = np.arange(x.shape[0], dtype=np.double)
        y = y - 2j*y[::-1]

        xi = np.array([0.5, 0.5])

        for method in ('nearest', 'linear', 'cubic'):
            p1 = griddata(x, y, xi, method=method)
            p2 = griddata(x, y, xi[None,:], method=method)
            assert_allclose(p1, p2, err_msg=method)

            xi1 = np.array([0.5])
            xi3 = np.array([0.5, 0.5, 0.5])
            assert_raises(ValueError, griddata, x, y, xi1,
                          method=method)
            assert_raises(ValueError, griddata, x, y, xi3,
                          method=method)


def test_nearest_options():
    # smoke test that NearestNDInterpolator accept cKDTree options
    npts, nd = 4, 3
    x = np.arange(npts*nd).reshape((npts, nd))
    y = np.arange(npts)
    nndi = NearestNDInterpolator(x, y)

    opts = {'balanced_tree': False, 'compact_nodes': False}
    nndi_o = NearestNDInterpolator(x, y, tree_options=opts)
    assert_allclose(nndi(x), nndi_o(x), atol=1e-14)


def test_nearest_list_argument():
    nd = np.array([[0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 1, 1],
                   [0, 0, 0, 0, 1, 1, 2]])
    d = nd[:, 3:]

    # z is np.array
    NI = NearestNDInterpolator((d[0], d[1]), d[2])
    assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])

    # z is list
    NI = NearestNDInterpolator((d[0], d[1]), list(d[2]))
    assert_array_equal(NI([0.1, 0.9], [0.1, 0.9]), [0, 2])
