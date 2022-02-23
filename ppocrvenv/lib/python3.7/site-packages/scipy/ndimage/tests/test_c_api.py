import numpy as np
from numpy.testing import assert_allclose

from scipy import ndimage
from scipy.ndimage import _ctest
from scipy.ndimage import _cytest
from scipy._lib._ccallback import LowLevelCallable

FILTER1D_FUNCTIONS = [
    lambda filter_size: _ctest.filter1d(filter_size),
    lambda filter_size: _cytest.filter1d(filter_size, with_signature=False),
    lambda filter_size: LowLevelCallable(_cytest.filter1d(filter_size, with_signature=True)),
    lambda filter_size: LowLevelCallable.from_cython(_cytest, "_filter1d",
                                                     _cytest.filter1d_capsule(filter_size)),
]

FILTER2D_FUNCTIONS = [
    lambda weights: _ctest.filter2d(weights),
    lambda weights: _cytest.filter2d(weights, with_signature=False),
    lambda weights: LowLevelCallable(_cytest.filter2d(weights, with_signature=True)),
    lambda weights: LowLevelCallable.from_cython(_cytest, "_filter2d", _cytest.filter2d_capsule(weights)),
]

TRANSFORM_FUNCTIONS = [
    lambda shift: _ctest.transform(shift),
    lambda shift: _cytest.transform(shift, with_signature=False),
    lambda shift: LowLevelCallable(_cytest.transform(shift, with_signature=True)),
    lambda shift: LowLevelCallable.from_cython(_cytest, "_transform", _cytest.transform_capsule(shift)),
]


def test_generic_filter():
    def filter2d(footprint_elements, weights):
        return (weights*footprint_elements).sum()

    def check(j):
        func = FILTER2D_FUNCTIONS[j]

        im = np.ones((20, 20))
        im[:10,:10] = 0
        footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        footprint_size = np.count_nonzero(footprint)
        weights = np.ones(footprint_size)/footprint_size

        res = ndimage.generic_filter(im, func(weights),
                                     footprint=footprint)
        std = ndimage.generic_filter(im, filter2d, footprint=footprint,
                                     extra_arguments=(weights,))
        assert_allclose(res, std, err_msg="#{} failed".format(j))

    for j, func in enumerate(FILTER2D_FUNCTIONS):
        check(j)


def test_generic_filter1d():
    def filter1d(input_line, output_line, filter_size):
        for i in range(output_line.size):
            output_line[i] = 0
            for j in range(filter_size):
                output_line[i] += input_line[i+j]
        output_line /= filter_size

    def check(j):
        func = FILTER1D_FUNCTIONS[j]

        im = np.tile(np.hstack((np.zeros(10), np.ones(10))), (10, 1))
        filter_size = 3

        res = ndimage.generic_filter1d(im, func(filter_size),
                                       filter_size)
        std = ndimage.generic_filter1d(im, filter1d, filter_size,
                                       extra_arguments=(filter_size,))
        assert_allclose(res, std, err_msg="#{} failed".format(j))

    for j, func in enumerate(FILTER1D_FUNCTIONS):
        check(j)


def test_geometric_transform():
    def transform(output_coordinates, shift):
        return output_coordinates[0] - shift, output_coordinates[1] - shift

    def check(j):
        func = TRANSFORM_FUNCTIONS[j]

        im = np.arange(12).reshape(4, 3).astype(np.float64)
        shift = 0.5

        res = ndimage.geometric_transform(im, func(shift))
        std = ndimage.geometric_transform(im, transform, extra_arguments=(shift,))
        assert_allclose(res, std, err_msg="#{} failed".format(j))

    for j, func in enumerate(TRANSFORM_FUNCTIONS):
        check(j)
