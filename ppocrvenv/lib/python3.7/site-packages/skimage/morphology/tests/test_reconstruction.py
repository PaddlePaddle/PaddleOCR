"""
These tests are originally part of CellProfiler, code licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from skimage._shared._warnings import expected_warnings
from skimage.morphology.grayreconstruct import reconstruction


def test_zeros():
    """Test reconstruction with image and mask of zeros"""
    assert_array_almost_equal(
        reconstruction(np.zeros((5, 7)), np.zeros((5, 7))), 0)


def test_image_equals_mask():
    """Test reconstruction where the image and mask are the same"""
    assert_array_almost_equal(
        reconstruction(np.ones((7, 5)), np.ones((7, 5))), 1)


def test_image_less_than_mask():
    """Test reconstruction where the image is uniform and less than mask"""
    image = np.ones((5, 5))
    mask = np.ones((5, 5)) * 2
    assert_array_almost_equal(reconstruction(image, mask), 1)


def test_one_image_peak():
    """Test reconstruction with one peak pixel"""
    image = np.ones((5, 5))
    image[2, 2] = 2
    mask = np.ones((5, 5)) * 3
    assert_array_almost_equal(reconstruction(image, mask), 2)


def test_two_image_peaks():
    """Test reconstruction with two peak pixels isolated by the mask"""
    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = np.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])

    expected = np.array([[2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [2, 2, 2, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3],
                         [1, 1, 1, 1, 1, 3, 3, 3]])
    assert_array_almost_equal(reconstruction(image, mask), expected)


def test_zero_image_one_mask():
    """Test reconstruction with an image of all zeros and a mask that's not"""
    result = reconstruction(np.zeros((10, 10)), np.ones((10, 10)))
    assert_array_almost_equal(result, 0)


def test_fill_hole():
    """Test reconstruction by erosion, which should fill holes in mask."""
    seed = np.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    result = reconstruction(seed, mask, method='erosion')
    assert_array_almost_equal(result, np.array([0, 3, 6, 4, 4, 4, 4, 4, 2, 0]))


def test_invalid_seed():
    seed = np.ones((5, 5))
    mask = np.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed * 2, mask,
                       method='dilation')
    with pytest.raises(ValueError):
        reconstruction(seed * 0.5, mask,
                       method='erosion')


def test_invalid_footprint():
    seed = np.ones((5, 5))
    mask = np.ones((5, 5))
    with pytest.raises(ValueError):
        reconstruction(seed, mask,
                       footprint=np.ones((4, 4)))
    with pytest.raises(ValueError):
        reconstruction(seed, mask,
                       footprint=np.ones((3, 4)))
    reconstruction(seed, mask, footprint=np.ones((3, 3)))


def test_invalid_method():
    seed = np.array([0, 8, 8, 8, 8, 8, 8, 8, 8, 0])
    mask = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    with pytest.raises(ValueError):
        reconstruction(seed, mask, method='foo')


def test_invalid_offset_not_none():
    """Test reconstruction with invalid not None offset parameter"""
    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 2, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 3, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])

    mask = np.array([[4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [4, 4, 4, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4],
                     [1, 1, 1, 1, 1, 4, 4, 4]])
    with pytest.raises(ValueError):
        reconstruction(image, mask, method='dilation',
                       footprint=np.ones((3, 3)), offset=np.array([3, 0]))


def test_offset_not_none():
    """Test reconstruction with valid offset parameter"""
    seed = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    mask = np.array([0, 8, 6, 8, 8, 8, 8, 4, 4, 0])
    expected = np.array([0, 3, 6, 6, 6, 6, 6, 4, 4, 0])

    assert_array_almost_equal(
        reconstruction(seed, mask, method='dilation',
                       footprint=np.ones(3), offset=np.array([0])), expected)


def test_deprecated_import():
    msg = "Importing from skimage.morphology.greyreconstruct is deprecated."
    with expected_warnings([msg + r"|\A\Z"]):
        from skimage.morphology.greyreconstruct import reconstruction


def test_selem_kwarg_deprecation():
    seed = np.array([0, 3, 6, 2, 1, 1, 1, 4, 2, 0])
    mask = np.array([0, 8, 6, 8, 8, 8, 8, 4, 4, 0])
    with expected_warnings(["`selem` is a deprecated argument name"]):
        reconstruction(seed, mask, method='dilation', selem=np.ones(3))
