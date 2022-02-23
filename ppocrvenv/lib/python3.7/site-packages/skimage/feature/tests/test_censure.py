import numpy as np
from skimage._shared.testing import assert_array_equal
from skimage.data import moon
from skimage.feature import CENSURE
from skimage._shared.testing import test_parallel
from skimage._shared import testing
from skimage.transform import rescale


img = moon()
np.random.seed(0)


def test_censure_on_rectangular_images():
    """Censure feature detector should work on 2D image of any shape."""
    rect_image = np.random.rand(300, 200)
    square_image = np.random.rand(200, 200)
    CENSURE().detect((square_image))
    CENSURE().detect((rect_image))


def test_keypoints_censure_color_image_unsupported_error():
    """Censure keypoints can be extracted from gray-scale images only."""
    with testing.raises(ValueError):
        CENSURE().detect(np.zeros((20, 20, 3)))


def test_keypoints_censure_mode_validity_error():
    """Mode argument in keypoints_censure can be either DoB, Octagon or
    STAR."""
    with testing.raises(ValueError):
        CENSURE(mode='dummy')


def test_keypoints_censure_scale_range_error():
    """Difference between the the max_scale and min_scale parameters in
    keypoints_censure should be greater than or equal to two."""
    with testing.raises(ValueError):
        CENSURE(min_scale=1, max_scale=2)


def test_keypoints_censure_moon_image_dob():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for DoB filter."""
    detector = CENSURE()
    detector.detect(img)
    expected_keypoints = np.array([[ 21, 497],
                                   [ 36,  46],
                                   [119, 350],
                                   [185, 177],
                                   [287, 250],
                                   [357, 239],
                                   [463, 116],
                                   [464, 132],
                                   [467, 260]])
    expected_scales = np.array([3, 4, 4, 2, 2, 3, 2, 2, 2])

    assert_array_equal(expected_keypoints, detector.keypoints)
    assert_array_equal(expected_scales, detector.scales)


@test_parallel()
def test_keypoints_censure_moon_image_octagon():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for Octagon filter."""

    detector = CENSURE(mode='octagon')
    # quarter scale image for speed
    detector.detect(rescale(img, 0.25,
                            anti_aliasing=False,
                            mode='constant'))
    expected_keypoints = np.array([[ 23,  27],
                                   [ 29,  89],
                                   [ 31,  87],
                                   [106,  59],
                                   [111,  67]])

    expected_scales = np.array([3, 2, 5, 2, 4])

    assert_array_equal(expected_keypoints, detector.keypoints)
    assert_array_equal(expected_scales, detector.scales)


def test_keypoints_censure_moon_image_star():
    """Verify the actual Censure keypoints and their corresponding scale with
    the expected values for STAR filter."""
    detector = CENSURE(mode='star')
    # quarter scale image for speed
    detector.detect(rescale(img, 0.25,
                            anti_aliasing=False,
                            mode='constant'))
    expected_keypoints = np.array([[ 23,  27],
                                   [ 29,  89],
                                   [ 30,  86],
                                   [107,  59],
                                   [109,  64],
                                   [111,  67],
                                   [113,  70]])

    expected_scales = np.array([3, 2, 4, 2, 5, 3, 2])

    assert_array_equal(expected_keypoints, detector.keypoints)
    assert_array_equal(expected_scales, detector.scales)
