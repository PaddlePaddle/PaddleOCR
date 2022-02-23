import numpy
from numpy.testing import (assert_, assert_equal, assert_array_equal,
                           assert_array_almost_equal)
import pytest
from pytest import raises as assert_raises

from scipy import ndimage

from . import types


class TestNdimageMorphology:

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_bf01(self, dtype):
        # brute force (bf) distance transform
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_bf(data, 'euclidean',
                                                return_indices=True)
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 2, 4, 2, 1, 0, 0],
                    [0, 0, 1, 4, 8, 4, 1, 0, 0],
                    [0, 0, 1, 2, 4, 2, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_almost_equal(out * out, expected)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 3, 2, 1, 2, 3, 3, 3],
                     [4, 4, 4, 4, 6, 4, 4, 4, 4],
                     [5, 5, 6, 6, 7, 6, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 1, 2, 4, 6, 7, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_bf02(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_bf(data, 'cityblock',
                                                return_indices=True)

        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 2, 2, 2, 1, 0, 0],
                    [0, 0, 1, 2, 3, 2, 1, 0, 0],
                    [0, 0, 1, 2, 2, 2, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_almost_equal(out, expected)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 3, 3, 1, 3, 3, 3, 3],
                     [4, 4, 4, 4, 7, 4, 4, 4, 4],
                     [5, 5, 6, 7, 7, 7, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 1, 1, 4, 7, 7, 7, 8],
                     [0, 1, 1, 1, 4, 7, 7, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(expected, ft)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_bf03(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_bf(data, 'chessboard',
                                                return_indices=True)

        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 2, 1, 1, 0, 0],
                    [0, 0, 1, 2, 2, 2, 1, 0, 0],
                    [0, 0, 1, 1, 2, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_almost_equal(out, expected)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 4, 2, 2, 2, 4, 3, 3],
                     [4, 4, 5, 6, 6, 6, 5, 4, 4],
                     [5, 5, 6, 6, 7, 6, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 5, 6, 6, 7, 8],
                     [0, 1, 1, 2, 6, 6, 7, 7, 8],
                     [0, 1, 1, 2, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 6, 6, 7, 7, 8],
                     [0, 1, 2, 4, 5, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_bf04(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        tdt, tft = ndimage.distance_transform_bf(data, return_indices=1)
        dts = []
        fts = []
        dt = numpy.zeros(data.shape, dtype=numpy.float64)
        ndimage.distance_transform_bf(data, distances=dt)
        dts.append(dt)
        ft = ndimage.distance_transform_bf(
            data, return_distances=False, return_indices=1)
        fts.append(ft)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_bf(
            data, return_distances=False, return_indices=True, indices=ft)
        fts.append(ft)
        dt, ft = ndimage.distance_transform_bf(
            data, return_indices=1)
        dts.append(dt)
        fts.append(ft)
        dt = numpy.zeros(data.shape, dtype=numpy.float64)
        ft = ndimage.distance_transform_bf(
            data, distances=dt, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        dt = ndimage.distance_transform_bf(
            data, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        dt = numpy.zeros(data.shape, dtype=numpy.float64)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_bf(
            data, distances=dt, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        for dt in dts:
            assert_array_almost_equal(tdt, dt)
        for ft in fts:
            assert_array_almost_equal(tft, ft)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_bf05(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_bf(
            data, 'euclidean', return_indices=True, sampling=[2, 2])
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4, 4, 4, 0, 0, 0],
                    [0, 0, 4, 8, 16, 8, 4, 0, 0],
                    [0, 0, 4, 16, 32, 16, 4, 0, 0],
                    [0, 0, 4, 8, 16, 8, 4, 0, 0],
                    [0, 0, 0, 4, 4, 4, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_almost_equal(out * out, expected)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 1, 2, 2, 2, 2],
                     [3, 3, 3, 2, 1, 2, 3, 3, 3],
                     [4, 4, 4, 4, 6, 4, 4, 4, 4],
                     [5, 5, 6, 6, 7, 6, 6, 5, 5],
                     [6, 6, 6, 7, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 1, 2, 4, 6, 7, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_bf06(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_bf(
            data, 'euclidean', return_indices=True, sampling=[2, 1])
        expected = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 4, 1, 0, 0, 0],
                    [0, 0, 1, 4, 8, 4, 1, 0, 0],
                    [0, 0, 1, 4, 9, 4, 1, 0, 0],
                    [0, 0, 1, 4, 8, 4, 1, 0, 0],
                    [0, 0, 0, 1, 4, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]
        assert_array_almost_equal(out * out, expected)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 2, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4, 4, 4, 4, 4],
                     [5, 5, 5, 5, 6, 5, 5, 5, 5],
                     [6, 6, 6, 6, 7, 6, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 6, 6, 6, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 1, 1, 7, 7, 7, 7, 8],
                     [0, 1, 1, 1, 6, 7, 7, 7, 8],
                     [0, 1, 2, 2, 4, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)

    def test_distance_transform_bf07(self):
        # test input validation per discussion on PR #13302
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        with assert_raises(RuntimeError):
            ndimage.distance_transform_bf(
                data, return_distances=False, return_indices=False
            )

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_cdt01(self, dtype):
        # chamfer type distance (cdt) transform
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_cdt(
            data, 'cityblock', return_indices=True)
        bf = ndimage.distance_transform_bf(data, 'cityblock')
        assert_array_almost_equal(bf, out)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 1, 1, 1, 2, 2, 2],
                     [3, 3, 2, 1, 1, 1, 2, 3, 3],
                     [4, 4, 4, 4, 1, 4, 4, 4, 4],
                     [5, 5, 5, 5, 7, 7, 6, 5, 5],
                     [6, 6, 6, 6, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 1, 1, 4, 7, 7, 7, 8],
                     [0, 1, 1, 1, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_cdt02(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_cdt(data, 'chessboard',
                                                 return_indices=True)
        bf = ndimage.distance_transform_bf(data, 'chessboard')
        assert_array_almost_equal(bf, out)

        expected = [[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [2, 2, 2, 1, 1, 1, 2, 2, 2],
                     [3, 3, 2, 2, 1, 2, 2, 3, 3],
                     [4, 4, 3, 2, 2, 2, 3, 4, 4],
                     [5, 5, 4, 6, 7, 6, 4, 5, 5],
                     [6, 6, 6, 6, 7, 7, 6, 6, 6],
                     [7, 7, 7, 7, 7, 7, 7, 7, 7],
                     [8, 8, 8, 8, 8, 8, 8, 8, 8]],
                    [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 2, 3, 4, 6, 7, 8],
                     [0, 1, 1, 2, 2, 6, 6, 7, 8],
                     [0, 1, 1, 1, 2, 6, 7, 7, 8],
                     [0, 1, 1, 2, 6, 6, 7, 7, 8],
                     [0, 1, 2, 2, 5, 6, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8]]]
        assert_array_almost_equal(ft, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_cdt03(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        tdt, tft = ndimage.distance_transform_cdt(data, return_indices=True)
        dts = []
        fts = []
        dt = numpy.zeros(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_cdt(data, distances=dt)
        dts.append(dt)
        ft = ndimage.distance_transform_cdt(
            data, return_distances=False, return_indices=True)
        fts.append(ft)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_cdt(
            data, return_distances=False, return_indices=True, indices=ft)
        fts.append(ft)
        dt, ft = ndimage.distance_transform_cdt(
            data, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        dt = numpy.zeros(data.shape, dtype=numpy.int32)
        ft = ndimage.distance_transform_cdt(
            data, distances=dt, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        dt = ndimage.distance_transform_cdt(
            data, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        dt = numpy.zeros(data.shape, dtype=numpy.int32)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_cdt(data, distances=dt,
                                       return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        for dt in dts:
            assert_array_almost_equal(tdt, dt)
        for ft in fts:
            assert_array_almost_equal(tft, ft)

    def test_distance_transform_cdt04(self):
        # test input validation per discussion on PR #13302
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        indices_out = numpy.zeros((data.ndim,) + data.shape, dtype=numpy.int32)
        with assert_raises(RuntimeError):
            ndimage.distance_transform_bf(
                data,
                return_distances=True,
                return_indices=False,
                indices=indices_out
            )

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_edt01(self, dtype):
        # euclidean distance transform (edt)
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out, ft = ndimage.distance_transform_edt(data, return_indices=True)
        bf = ndimage.distance_transform_bf(data, 'euclidean')
        assert_array_almost_equal(bf, out)

        dt = ft - numpy.indices(ft.shape[1:], dtype=ft.dtype)
        dt = dt.astype(numpy.float64)
        numpy.multiply(dt, dt, dt)
        dt = numpy.add.reduce(dt, axis=0)
        numpy.sqrt(dt, dt)

        assert_array_almost_equal(bf, dt)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_edt02(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        tdt, tft = ndimage.distance_transform_edt(data, return_indices=True)
        dts = []
        fts = []
        dt = numpy.zeros(data.shape, dtype=numpy.float64)
        ndimage.distance_transform_edt(data, distances=dt)
        dts.append(dt)
        ft = ndimage.distance_transform_edt(
            data, return_distances=0, return_indices=True)
        fts.append(ft)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_edt(
            data, return_distances=False, return_indices=True, indices=ft)
        fts.append(ft)
        dt, ft = ndimage.distance_transform_edt(
            data, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        dt = numpy.zeros(data.shape, dtype=numpy.float64)
        ft = ndimage.distance_transform_edt(
            data, distances=dt, return_indices=True)
        dts.append(dt)
        fts.append(ft)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        dt = ndimage.distance_transform_edt(
            data, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        dt = numpy.zeros(data.shape, dtype=numpy.float64)
        ft = numpy.indices(data.shape, dtype=numpy.int32)
        ndimage.distance_transform_edt(
            data, distances=dt, return_indices=True, indices=ft)
        dts.append(dt)
        fts.append(ft)
        for dt in dts:
            assert_array_almost_equal(tdt, dt)
        for ft in fts:
            assert_array_almost_equal(tft, ft)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_edt03(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        ref = ndimage.distance_transform_bf(data, 'euclidean', sampling=[2, 2])
        out = ndimage.distance_transform_edt(data, sampling=[2, 2])
        assert_array_almost_equal(ref, out)

    @pytest.mark.parametrize('dtype', types)
    def test_distance_transform_edt4(self, dtype):
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        ref = ndimage.distance_transform_bf(data, 'euclidean', sampling=[2, 1])
        out = ndimage.distance_transform_edt(data, sampling=[2, 1])
        assert_array_almost_equal(ref, out)

    def test_distance_transform_edt5(self):
        # Ticket #954 regression test
        out = ndimage.distance_transform_edt(False)
        assert_array_almost_equal(out, [0.])

    def test_distance_transform_edt6(self):
        # test input validation per discussion on PR #13302
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        distances_out = numpy.zeros(data.shape, dtype=numpy.float64)
        with assert_raises(RuntimeError):
            ndimage.distance_transform_bf(
                data,
                return_indices=True,
                return_distances=False,
                distances=distances_out
            )

    def test_generate_structure01(self):
        struct = ndimage.generate_binary_structure(0, 1)
        assert_array_almost_equal(struct, 1)

    def test_generate_structure02(self):
        struct = ndimage.generate_binary_structure(1, 1)
        assert_array_almost_equal(struct, [1, 1, 1])

    def test_generate_structure03(self):
        struct = ndimage.generate_binary_structure(2, 1)
        assert_array_almost_equal(struct, [[0, 1, 0],
                                           [1, 1, 1],
                                           [0, 1, 0]])

    def test_generate_structure04(self):
        struct = ndimage.generate_binary_structure(2, 2)
        assert_array_almost_equal(struct, [[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]])

    def test_iterate_structure01(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        out = ndimage.iterate_structure(struct, 2)
        assert_array_almost_equal(out, [[0, 0, 1, 0, 0],
                                        [0, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 1],
                                        [0, 1, 1, 1, 0],
                                        [0, 0, 1, 0, 0]])

    def test_iterate_structure02(self):
        struct = [[0, 1],
                  [1, 1],
                  [0, 1]]
        out = ndimage.iterate_structure(struct, 2)
        assert_array_almost_equal(out, [[0, 0, 1],
                                        [0, 1, 1],
                                        [1, 1, 1],
                                        [0, 1, 1],
                                        [0, 0, 1]])

    def test_iterate_structure03(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        out = ndimage.iterate_structure(struct, 2, 1)
        expected = [[0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0]]
        assert_array_almost_equal(out[0], expected)
        assert_equal(out[1], [2, 2])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion01(self, dtype):
        data = numpy.ones([], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion02(self, dtype):
        data = numpy.ones([], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion03(self, dtype):
        data = numpy.ones([1], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion04(self, dtype):
        data = numpy.ones([1], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion05(self, dtype):
        data = numpy.ones([3], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion06(self, dtype):
        data = numpy.ones([3], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion07(self, dtype):
        data = numpy.ones([5], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [0, 1, 1, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion08(self, dtype):
        data = numpy.ones([5], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [1, 1, 1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion09(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [0, 0, 0, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion10(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [1, 0, 0, 0, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion11(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        struct = [1, 0, 1]
        out = ndimage.binary_erosion(data, struct, border_value=1)
        assert_array_almost_equal(out, [1, 0, 1, 0, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion12(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        struct = [1, 0, 1]
        out = ndimage.binary_erosion(data, struct, border_value=1, origin=-1)
        assert_array_almost_equal(out, [0, 1, 0, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion13(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        struct = [1, 0, 1]
        out = ndimage.binary_erosion(data, struct, border_value=1, origin=1)
        assert_array_almost_equal(out, [1, 1, 0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion14(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        struct = [1, 1]
        out = ndimage.binary_erosion(data, struct, border_value=1)
        assert_array_almost_equal(out, [1, 1, 0, 0, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion15(self, dtype):
        data = numpy.ones([5], dtype)
        data[2] = 0
        struct = [1, 1]
        out = ndimage.binary_erosion(data, struct, border_value=1, origin=-1)
        assert_array_almost_equal(out, [1, 0, 0, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion16(self, dtype):
        data = numpy.ones([1, 1], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [[1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion17(self, dtype):
        data = numpy.ones([1, 1], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [[0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion18(self, dtype):
        data = numpy.ones([1, 3], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [[0, 0, 0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion19(self, dtype):
        data = numpy.ones([1, 3], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [[1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion20(self, dtype):
        data = numpy.ones([3, 3], dtype)
        out = ndimage.binary_erosion(data)
        assert_array_almost_equal(out, [[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion21(self, dtype):
        data = numpy.ones([3, 3], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion22(self, dtype):
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_erosion(data, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion23(self, dtype):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_erosion(data, struct, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion24(self, dtype):
        struct = [[0, 1],
                  [1, 1]]
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_erosion(data, struct, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion25(self, dtype):
        struct = [[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 0, 1, 1],
                            [0, 0, 1, 0, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_erosion(data, struct, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_erosion26(self, dtype):
        struct = [[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 0, 1, 1],
                            [0, 0, 1, 0, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_erosion(data, struct, border_value=1,
                                     origin=(-1, -1))
        assert_array_almost_equal(out, expected)

    def test_binary_erosion27(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_erosion(data, struct, border_value=1,
                                     iterations=2)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion28(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=2, output=out)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion29(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], bool)
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, iterations=3)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion30(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=3, output=out)
        assert_array_almost_equal(out, expected)

        # test with output memory overlap
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=3, output=data)
        assert_array_almost_equal(data, expected)

    def test_binary_erosion31(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1]]
        data = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=1, output=out, origin=(-1, -1))
        assert_array_almost_equal(out, expected)

    def test_binary_erosion32(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, iterations=2)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion33(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        mask = [[1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1]]
        data = numpy.array([[0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 1, 0, 0, 1],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, mask=mask, iterations=-1)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion34(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        mask = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_erosion(data, struct,
                                     border_value=1, mask=mask)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion35(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        mask = [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], bool)
        tmp = [[0, 0, 1, 0, 0, 0, 0],
               [0, 1, 1, 1, 0, 0, 0],
               [1, 1, 1, 1, 1, 0, 1],
               [0, 1, 1, 1, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 1]]
        expected = numpy.logical_and(tmp, mask)
        tmp = numpy.logical_and(data, numpy.logical_not(mask))
        expected = numpy.logical_or(expected, tmp)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=1, output=out,
                               origin=(-1, -1), mask=mask)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion36(self):
        struct = [[0, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0]]
        mask = [[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]
        tmp = [[0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 1],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 0, 1, 1],
                            [0, 0, 1, 0, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]])
        expected = numpy.logical_and(tmp, mask)
        tmp = numpy.logical_and(data, numpy.logical_not(mask))
        expected = numpy.logical_or(expected, tmp)
        out = ndimage.binary_erosion(data, struct, mask=mask,
                                     border_value=1, origin=(-1, -1))
        assert_array_almost_equal(out, expected)

    def test_binary_erosion37(self):
        a = numpy.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]], dtype=bool)
        b = numpy.zeros_like(a)
        out = ndimage.binary_erosion(a, structure=a, output=b, iterations=0,
                                     border_value=True, brute_force=True)
        assert_(out is b)
        assert_array_equal(
            ndimage.binary_erosion(a, structure=a, iterations=0,
                                   border_value=True),
            b)

    def test_binary_erosion38(self):
        data = numpy.array([[1, 0, 1],
                           [0, 1, 0],
                           [1, 0, 1]], dtype=bool)
        iterations = 2.0
        with assert_raises(TypeError):
            _ = ndimage.binary_erosion(data, iterations=iterations)

    def test_binary_erosion39(self):
        iterations = numpy.int32(3)
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=iterations, output=out)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion40(self):
        iterations = numpy.int64(3)
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_erosion(data, struct, border_value=1,
                               iterations=iterations, output=out)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation01(self, dtype):
        data = numpy.ones([], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, 1)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation02(self, dtype):
        data = numpy.zeros([], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, 0)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation03(self, dtype):
        data = numpy.ones([1], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation04(self, dtype):
        data = numpy.zeros([1], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation05(self, dtype):
        data = numpy.ones([3], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation06(self, dtype):
        data = numpy.zeros([3], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [0, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation07(self, dtype):
        data = numpy.zeros([3], dtype)
        data[1] = 1
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation08(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        data[3] = 1
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [1, 1, 1, 1, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation09(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [1, 1, 1, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation10(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        out = ndimage.binary_dilation(data, origin=-1)
        assert_array_almost_equal(out, [0, 1, 1, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation11(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        out = ndimage.binary_dilation(data, origin=1)
        assert_array_almost_equal(out, [1, 1, 0, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation12(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        struct = [1, 0, 1]
        out = ndimage.binary_dilation(data, struct)
        assert_array_almost_equal(out, [1, 0, 1, 0, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation13(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        struct = [1, 0, 1]
        out = ndimage.binary_dilation(data, struct, border_value=1)
        assert_array_almost_equal(out, [1, 0, 1, 0, 1])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation14(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        struct = [1, 0, 1]
        out = ndimage.binary_dilation(data, struct, origin=-1)
        assert_array_almost_equal(out, [0, 1, 0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation15(self, dtype):
        data = numpy.zeros([5], dtype)
        data[1] = 1
        struct = [1, 0, 1]
        out = ndimage.binary_dilation(data, struct,
                                      origin=-1, border_value=1)
        assert_array_almost_equal(out, [1, 1, 0, 1, 0])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation16(self, dtype):
        data = numpy.ones([1, 1], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [[1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation17(self, dtype):
        data = numpy.zeros([1, 1], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [[0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation18(self, dtype):
        data = numpy.ones([1, 3], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [[1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation19(self, dtype):
        data = numpy.ones([3, 3], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation20(self, dtype):
        data = numpy.zeros([3, 3], dtype)
        data[1, 1] = 1
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, [[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation21(self, dtype):
        struct = ndimage.generate_binary_structure(2, 2)
        data = numpy.zeros([3, 3], dtype)
        data[1, 1] = 1
        out = ndimage.binary_dilation(data, struct)
        assert_array_almost_equal(out, [[1, 1, 1],
                                        [1, 1, 1],
                                        [1, 1, 1]])

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation22(self, dtype):
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation23(self, dtype):
        expected = [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 0, 1],
                    [1, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation24(self, dtype):
        expected = [[1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, origin=(1, 1))
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation25(self, dtype):
        expected = [[1, 1, 0, 0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, 0, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, origin=(1, 1), border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation26(self, dtype):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [[1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, struct)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation27(self, dtype):
        struct = [[0, 1],
                  [1, 1]]
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, struct)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation28(self, dtype):
        expected = [[1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]]
        data = numpy.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, border_value=1)
        assert_array_almost_equal(out, expected)

    def test_binary_dilation29(self):
        struct = [[0, 1],
                  [1, 1]]
        expected = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]

        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_dilation(data, struct, iterations=2)
        assert_array_almost_equal(out, expected)

    def test_binary_dilation30(self):
        struct = [[0, 1],
                  [1, 1]]
        expected = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]

        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_dilation(data, struct, iterations=2, output=out)
        assert_array_almost_equal(out, expected)

    def test_binary_dilation31(self):
        struct = [[0, 1],
                  [1, 1]]
        expected = [[0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]

        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_dilation(data, struct, iterations=3)
        assert_array_almost_equal(out, expected)

    def test_binary_dilation32(self):
        struct = [[0, 1],
                  [1, 1]]
        expected = [[0, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0]]

        data = numpy.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0]], bool)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_dilation(data, struct, iterations=3, output=out)
        assert_array_almost_equal(out, expected)

    def test_binary_dilation33(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0, 0],
                                [0, 1, 1, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        mask = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)

        out = ndimage.binary_dilation(data, struct, iterations=-1,
                                      mask=mask, border_value=0)
        assert_array_almost_equal(out, expected)

    def test_binary_dilation34(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        mask = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.zeros(mask.shape, bool)
        out = ndimage.binary_dilation(data, struct, iterations=-1,
                                      mask=mask, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_dilation35(self, dtype):
        tmp = [[1, 1, 0, 0, 0, 0, 1, 1],
               [1, 0, 0, 0, 1, 0, 1, 1],
               [0, 0, 1, 1, 1, 1, 1, 1],
               [0, 1, 1, 1, 1, 0, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [0, 1, 0, 0, 1, 0, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1]]
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]])
        mask = [[0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0]]
        expected = numpy.logical_and(tmp, mask)
        tmp = numpy.logical_and(data, numpy.logical_not(mask))
        expected = numpy.logical_or(expected, tmp)
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_dilation(data, mask=mask,
                                      origin=(1, 1), border_value=1)
        assert_array_almost_equal(out, expected)

    def test_binary_propagation01(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0, 0],
                                [0, 1, 1, 0, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        mask = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)

        out = ndimage.binary_propagation(data, struct,
                                         mask=mask, border_value=0)
        assert_array_almost_equal(out, expected)

    def test_binary_propagation02(self):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        mask = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.zeros(mask.shape, bool)
        out = ndimage.binary_propagation(data, struct,
                                         mask=mask, border_value=1)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_opening01(self, dtype):
        expected = [[0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 0, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_opening(data)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_opening02(self, dtype):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [[1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_opening(data, struct)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_closing01(self, dtype):
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 0, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_closing(data)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize('dtype', types)
    def test_binary_closing02(self, dtype):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_closing(data, struct)
        assert_array_almost_equal(out, expected)

    def test_binary_fill_holes01(self):
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_binary_fill_holes02(self):
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_binary_fill_holes03(self):
        expected = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0, 1, 1, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1],
                                [0, 1, 1, 1, 0, 1, 1, 1],
                                [0, 0, 1, 0, 0, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        data = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 1, 1, 1],
                            [0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 0, 1, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0]], bool)
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_grey_erosion01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        output = ndimage.grey_erosion(array, footprint=footprint)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], output)

    def test_grey_erosion01_overlap(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        ndimage.grey_erosion(array, footprint=footprint, output=array)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], array)

    def test_grey_erosion02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        output = ndimage.grey_erosion(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal([[2, 2, 1, 1, 1],
                                   [2, 3, 1, 3, 1],
                                   [5, 5, 3, 3, 1]], output)

    def test_grey_erosion03(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.grey_erosion(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal([[1, 1, 0, 0, 0],
                                   [1, 2, 0, 2, 0],
                                   [4, 4, 2, 2, 0]], output)

    def test_grey_dilation01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[0, 1, 1], [1, 0, 1]]
        output = ndimage.grey_dilation(array, footprint=footprint)
        assert_array_almost_equal([[7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7],
                                   [8, 8, 8, 7, 7]], output)

    def test_grey_dilation02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[0, 1, 1], [1, 0, 1]]
        structure = [[0, 0, 0], [0, 0, 0]]
        output = ndimage.grey_dilation(array, footprint=footprint,
                                       structure=structure)
        assert_array_almost_equal([[7, 7, 9, 9, 5],
                                   [7, 9, 8, 9, 7],
                                   [8, 8, 8, 7, 7]], output)

    def test_grey_dilation03(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[0, 1, 1], [1, 0, 1]]
        structure = [[1, 1, 1], [1, 1, 1]]
        output = ndimage.grey_dilation(array, footprint=footprint,
                                       structure=structure)
        assert_array_almost_equal([[8, 8, 10, 10, 6],
                                   [8, 10, 9, 10, 8],
                                   [9, 9, 9, 8, 8]], output)

    def test_grey_opening01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        tmp = ndimage.grey_erosion(array, footprint=footprint)
        expected = ndimage.grey_dilation(tmp, footprint=footprint)
        output = ndimage.grey_opening(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_grey_opening02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp = ndimage.grey_erosion(array, footprint=footprint,
                                   structure=structure)
        expected = ndimage.grey_dilation(tmp, footprint=footprint,
                                         structure=structure)
        output = ndimage.grey_opening(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal(expected, output)

    def test_grey_closing01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        tmp = ndimage.grey_dilation(array, footprint=footprint)
        expected = ndimage.grey_erosion(tmp, footprint=footprint)
        output = ndimage.grey_closing(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_grey_closing02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp = ndimage.grey_dilation(array, footprint=footprint,
                                    structure=structure)
        expected = ndimage.grey_erosion(tmp, footprint=footprint,
                                        structure=structure)
        output = ndimage.grey_closing(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal(expected, output)

    def test_morphological_gradient01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        expected = tmp1 - tmp2
        output = numpy.zeros(array.shape, array.dtype)
        ndimage.morphological_gradient(array, footprint=footprint,
                                       structure=structure, output=output)
        assert_array_almost_equal(expected, output)

    def test_morphological_gradient02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        expected = tmp1 - tmp2
        output = ndimage.morphological_gradient(array, footprint=footprint,
                                                structure=structure)
        assert_array_almost_equal(expected, output)

    def test_morphological_laplace01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        expected = tmp1 + tmp2 - 2 * array
        output = numpy.zeros(array.shape, array.dtype)
        ndimage.morphological_laplace(array, footprint=footprint,
                                      structure=structure, output=output)
        assert_array_almost_equal(expected, output)

    def test_morphological_laplace02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp1 = ndimage.grey_dilation(array, footprint=footprint,
                                     structure=structure)
        tmp2 = ndimage.grey_erosion(array, footprint=footprint,
                                    structure=structure)
        expected = tmp1 + tmp2 - 2 * array
        output = ndimage.morphological_laplace(array, footprint=footprint,
                                               structure=structure)
        assert_array_almost_equal(expected, output)

    def test_white_tophat01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp = ndimage.grey_opening(array, footprint=footprint,
                                   structure=structure)
        expected = array - tmp
        output = numpy.zeros(array.shape, array.dtype)
        ndimage.white_tophat(array, footprint=footprint,
                             structure=structure, output=output)
        assert_array_almost_equal(expected, output)

    def test_white_tophat02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp = ndimage.grey_opening(array, footprint=footprint,
                                   structure=structure)
        expected = array - tmp
        output = ndimage.white_tophat(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal(expected, output)

    def test_white_tophat03(self):
        array = numpy.array([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0, 1, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]], dtype=numpy.bool_)
        structure = numpy.ones((3, 3), dtype=numpy.bool_)
        expected = numpy.array([[0, 1, 1, 0, 0, 0, 0],
                                [1, 0, 0, 1, 1, 1, 0],
                                [1, 0, 0, 1, 1, 1, 0],
                                [0, 1, 1, 0, 0, 0, 1],
                                [0, 1, 1, 0, 1, 0, 1],
                                [0, 1, 1, 0, 0, 0, 1],
                                [0, 0, 0, 1, 1, 1, 1]], dtype=numpy.bool_)

        output = ndimage.white_tophat(array, structure=structure)
        assert_array_equal(expected, output)

    def test_white_tophat04(self):
        array = numpy.eye(5, dtype=numpy.bool_)
        structure = numpy.ones((3, 3), dtype=numpy.bool_)

        # Check that type mismatch is properly handled
        output = numpy.empty_like(array, dtype=numpy.float64)
        ndimage.white_tophat(array, structure=structure, output=output)

    def test_black_tophat01(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp = ndimage.grey_closing(array, footprint=footprint,
                                   structure=structure)
        expected = tmp - array
        output = numpy.zeros(array.shape, array.dtype)
        ndimage.black_tophat(array, footprint=footprint,
                             structure=structure, output=output)
        assert_array_almost_equal(expected, output)

    def test_black_tophat02(self):
        array = numpy.array([[3, 2, 5, 1, 4],
                             [7, 6, 9, 3, 5],
                             [5, 8, 3, 7, 1]])
        footprint = [[1, 0, 1], [1, 1, 0]]
        structure = [[0, 0, 0], [0, 0, 0]]
        tmp = ndimage.grey_closing(array, footprint=footprint,
                                   structure=structure)
        expected = tmp - array
        output = ndimage.black_tophat(array, footprint=footprint,
                                      structure=structure)
        assert_array_almost_equal(expected, output)

    def test_black_tophat03(self):
        array = numpy.array([[1, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 0, 1, 0],
                             [0, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 1]], dtype=numpy.bool_)
        structure = numpy.ones((3, 3), dtype=numpy.bool_)
        expected = numpy.array([[0, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 0]], dtype=numpy.bool_)

        output = ndimage.black_tophat(array, structure=structure)
        assert_array_equal(expected, output)

    def test_black_tophat04(self):
        array = numpy.eye(5, dtype=numpy.bool_)
        structure = numpy.ones((3, 3), dtype=numpy.bool_)

        # Check that type mismatch is properly handled
        output = numpy.empty_like(array, dtype=numpy.float64)
        ndimage.black_tophat(array, structure=structure, output=output)

    @pytest.mark.parametrize('dtype', types)
    def test_hit_or_miss01(self, dtype):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]
        data = numpy.array([[0, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1],
                            [0, 0, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0]], dtype)
        out = numpy.zeros(data.shape, bool)
        ndimage.binary_hit_or_miss(data, struct, output=out)
        assert_array_almost_equal(expected, out)

    @pytest.mark.parametrize('dtype', types)
    def test_hit_or_miss02(self, dtype):
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
        expected = [[0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 1, 0, 0, 1, 1, 1, 0],
                            [1, 1, 1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_hit_or_miss(data, struct)
        assert_array_almost_equal(expected, out)

    @pytest.mark.parametrize('dtype', types)
    def test_hit_or_miss03(self, dtype):
        struct1 = [[0, 0, 0],
                   [1, 1, 1],
                   [0, 0, 0]]
        struct2 = [[1, 1, 1],
                   [0, 0, 0],
                   [1, 1, 1]]
        expected = [[0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
        data = numpy.array([[0, 1, 0, 0, 1, 1, 1, 0],
                            [1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 0, 1, 1, 0],
                            [0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0]], dtype)
        out = ndimage.binary_hit_or_miss(data, struct1, struct2)
        assert_array_almost_equal(expected, out)


class TestDilateFix:

    def setup_method(self):
        # dilation related setup
        self.array = numpy.array([[0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 1, 1, 0],
                                  [0, 0, 0, 0, 0]], dtype=numpy.uint8)

        self.sq3x3 = numpy.ones((3, 3))
        dilated3x3 = ndimage.binary_dilation(self.array, structure=self.sq3x3)
        self.dilated3x3 = dilated3x3.view(numpy.uint8)

    def test_dilation_square_structure(self):
        result = ndimage.grey_dilation(self.array, structure=self.sq3x3)
        # +1 accounts for difference between grey and binary dilation
        assert_array_almost_equal(result, self.dilated3x3 + 1)

    def test_dilation_scalar_size(self):
        result = ndimage.grey_dilation(self.array, size=3)
        assert_array_almost_equal(result, self.dilated3x3)


class TestBinaryOpeningClosing:

    def setup_method(self):
        a = numpy.zeros((5, 5), dtype=bool)
        a[1:4, 1:4] = True
        a[4, 4] = True
        self.array = a
        self.sq3x3 = numpy.ones((3, 3))
        self.opened_old = ndimage.binary_opening(self.array, self.sq3x3,
                                                 1, None, 0)
        self.closed_old = ndimage.binary_closing(self.array, self.sq3x3,
                                                 1, None, 0)

    def test_opening_new_arguments(self):
        opened_new = ndimage.binary_opening(self.array, self.sq3x3, 1, None,
                                            0, None, 0, False)
        assert_array_equal(opened_new, self.opened_old)

    def test_closing_new_arguments(self):
        closed_new = ndimage.binary_closing(self.array, self.sq3x3, 1, None,
                                            0, None, 0, False)
        assert_array_equal(closed_new, self.closed_old)


def test_binary_erosion_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = numpy.ones([1])
    assert_raises(TypeError, ndimage.binary_erosion, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_erosion, data, iterations=1.5)


def test_binary_dilation_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = numpy.ones([1])
    assert_raises(TypeError, ndimage.binary_dilation, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_dilation, data, iterations=1.5)


def test_binary_opening_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = numpy.ones([1])
    assert_raises(TypeError, ndimage.binary_opening, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_opening, data, iterations=1.5)


def test_binary_closing_noninteger_iterations():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = numpy.ones([1])
    assert_raises(TypeError, ndimage.binary_closing, data, iterations=0.5)
    assert_raises(TypeError, ndimage.binary_closing, data, iterations=1.5)


def test_binary_closing_noninteger_brute_force_passes_when_true():
    # regression test for gh-9905, gh-9909: ValueError for
    # non integer iterations
    data = numpy.ones([1])

    assert ndimage.binary_erosion(
        data, iterations=2, brute_force=1.5
    ) == ndimage.binary_erosion(data, iterations=2, brute_force=bool(1.5))
    assert ndimage.binary_erosion(
        data, iterations=2, brute_force=0.0
    ) == ndimage.binary_erosion(data, iterations=2, brute_force=bool(0.0))
