import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from skimage import color, data, draw, feature, img_as_float
from skimage._shared import filters
from skimage._shared._warnings import expected_warnings
from skimage._shared.testing import fetch
from skimage._shared.utils import _supported_float_type


def test_hog_output_size():
    img = img_as_float(data.astronaut()[:256, :].mean(axis=2))

    fd = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1), block_norm='L1')

    assert len(fd) == 9 * (256 // 8) * (512 // 8)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_hog_output_correctness_l1_norm(dtype):
    img = color.rgb2gray(data.astronaut()).astype(dtype=dtype, copy=False)
    correct_output = np.load(fetch('data/astronaut_GRAY_hog_L1.npy'))

    output = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L1',
                         feature_vector=True, transform_sqrt=False,
                         visualize=False)
    float_dtype = _supported_float_type(dtype)
    assert output.dtype == float_dtype
    decimal = 7 if float_dtype == np.float64 else 5
    assert_almost_equal(output, correct_output, decimal=decimal)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_hog_output_correctness_l2hys_norm(dtype):
    img = color.rgb2gray(data.astronaut()).astype(dtype=dtype, copy=False)
    correct_output = np.load(fetch('data/astronaut_GRAY_hog_L2-Hys.npy'))

    output = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(3, 3), block_norm='L2-Hys',
                         feature_vector=True, transform_sqrt=False,
                         visualize=False)
    float_dtype = _supported_float_type(dtype)
    assert output.dtype == float_dtype
    decimal = 7 if float_dtype == np.float64 else 5
    assert_almost_equal(output, correct_output, decimal=decimal)


def test_hog_image_size_cell_size_mismatch():
    image = data.camera()[:150, :200]
    fd = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(1, 1), block_norm='L1')
    assert len(fd) == 9 * (150 // 8) * (200 // 8)


def test_hog_odd_cell_size():
    img = np.zeros((3, 3))
    img[2, 2] = 1

    correct_output = np.zeros((9, ))
    correct_output[0] = 0.5
    correct_output[4] = 0.5

    output = feature.hog(img, pixels_per_cell=(3, 3),
                         cells_per_block=(1, 1), block_norm='L1')

    assert_almost_equal(output, correct_output, decimal=1)


def test_hog_basic_orientations_and_data_types():
    # scenario:
    #  1) create image (with float values) where upper half is filled by
    #     zeros, bottom half by 100
    #  2) create unsigned integer version of this image
    #  3) calculate feature.hog() for both images, both with 'transform_sqrt'
    #     option enabled and disabled
    #  4) verify that all results are equal where expected
    #  5) verify that computed feature vector is as expected
    #  6) repeat the scenario for 90, 180 and 270 degrees rotated images

    # size of testing image
    width = height = 35

    image0 = np.zeros((height, width), dtype='float')
    image0[height // 2:] = 100

    for rot in range(4):
        # rotate by 0, 90, 180 and 270 degrees
        image_float = np.rot90(image0, rot)

        # create uint8 image from image_float
        image_uint8 = image_float.astype('uint8')

        (hog_float, hog_img_float) = feature.hog(
            image_float, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualize=True, transform_sqrt=False,
            block_norm='L1')
        (hog_uint8, hog_img_uint8) = feature.hog(
            image_uint8, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualize=True, transform_sqrt=False,
            block_norm='L1')
        (hog_float_norm, hog_img_float_norm) = feature.hog(
            image_float, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualize=True, transform_sqrt=True,
            block_norm='L1')
        (hog_uint8_norm, hog_img_uint8_norm) = feature.hog(
            image_uint8, orientations=4, pixels_per_cell=(8, 8),
            cells_per_block=(1, 1), visualize=True, transform_sqrt=True,
            block_norm='L1')

        # set to True to enable manual debugging with graphical output,
        # must be False for automatic testing
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(image_float)
            plt.colorbar()
            plt.title('image')
            plt.subplot(2, 3, 2)
            plt.imshow(hog_img_float)
            plt.colorbar()
            plt.title('HOG result visualisation (float img)')
            plt.subplot(2, 3, 5)
            plt.imshow(hog_img_uint8)
            plt.colorbar()
            plt.title('HOG result visualisation (uint8 img)')
            plt.subplot(2, 3, 3)
            plt.imshow(hog_img_float_norm)
            plt.colorbar()
            plt.title('HOG result (transform_sqrt) visualisation (float img)')
            plt.subplot(2, 3, 6)
            plt.imshow(hog_img_uint8_norm)
            plt.colorbar()
            plt.title('HOG result (transform_sqrt) visualisation (uint8 img)')
            plt.show()

        # results (features and visualisation) for float and uint8 images must
        # be almost equal
        assert_almost_equal(hog_float, hog_uint8)
        assert_almost_equal(hog_img_float, hog_img_uint8)

        # resulting features should be almost equal
        # when 'transform_sqrt' is enabled
        # or disabled (for current simple testing image)
        assert_almost_equal(hog_float, hog_float_norm, decimal=4)
        assert_almost_equal(hog_float, hog_uint8_norm, decimal=4)

        # reshape resulting feature vector to matrix with 4 columns (each
        # corresponding to one of 4 directions); only one direction should
        # contain nonzero values (this is manually determined for testing
        # image)
        actual = np.max(hog_float.reshape(-1, 4), axis=0)

        if rot in [0, 2]:
            # image is rotated by 0 and 180 degrees
            desired = [0, 0, 1, 0]
        elif rot in [1, 3]:
            # image is rotated by 90 and 270 degrees
            desired = [1, 0, 0, 0]
        else:
            raise Exception('Result is not determined for this rotation.')

        assert_almost_equal(actual, desired, decimal=2)


def test_hog_orientations_circle():
    # scenario:
    #  1) create image with blurred circle in the middle
    #  2) calculate feature.hog()
    #  3) verify that the resulting feature vector contains uniformly
    #     distributed values for all orientations, i.e. no orientation is
    #     lost or emphasized
    #  4) repeat the scenario for other 'orientations' option

    # size of testing image
    width = height = 100

    image = np.zeros((height, width))
    rr, cc = draw.disk((int(height / 2), int(width / 2)), int(width / 3))
    image[rr, cc] = 100
    image = filters.gaussian(image, 2, mode='reflect')

    for orientations in range(2, 15):
        (hog, hog_img) = feature.hog(image, orientations=orientations,
                                     pixels_per_cell=(8, 8),
                                     cells_per_block=(1, 1), visualize=True,
                                     transform_sqrt=False,
                                     block_norm='L1')

        # set to True to enable manual debugging with graphical output,
        # must be False for automatic testing
        if False:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.colorbar()
            plt.title('image_float')
            plt.subplot(1, 2, 2)
            plt.imshow(hog_img)
            plt.colorbar()
            plt.title('HOG result visualisation, '
                      'orientations=%d' % (orientations))
            plt.show()

        # reshape resulting feature vector to matrix with N columns (each
        # column corresponds to one direction),
        hog_matrix = hog.reshape(-1, orientations)

        # compute mean values in the resulting feature vector for each
        # direction, these values should be almost equal to the global mean
        # value (since the image contains a circle), i.e., all directions have
        # same contribution to the result
        actual = np.mean(hog_matrix, axis=0)
        desired = np.mean(hog_matrix)
        assert_almost_equal(actual, desired, decimal=1)


def test_hog_visualization_orientation():
    """Test that the visualization produces a line with correct orientation

    The hog visualization is expected to draw line segments perpendicular to
    the midpoints of orientation bins.  This example verifies that when
    orientations=3 and the gradient is entirely in the middle bin (bisected
    by the y-axis), the line segment drawn by the visualization is horizontal.
    """

    width = height = 11

    image = np.zeros((height, width), dtype='float')
    image[height // 2:] = 1

    _, hog_image = feature.hog(
        image,
        orientations=3,
        pixels_per_cell=(width, height),
        cells_per_block=(1, 1),
        visualize=True,
        block_norm='L1'
    )

    middle_index = height // 2
    indices_excluding_middle = [x for x in range(height) if x != middle_index]

    assert (hog_image[indices_excluding_middle, :] == 0).all()
    assert (hog_image[middle_index, 1:-1] > 0).all()


def test_hog_block_normalization_incorrect_error():
    img = np.eye(4)
    with pytest.raises(ValueError):
        feature.hog(img, block_norm='Linf')


@pytest.mark.parametrize("shape,multichannel", [
    ((3, 3, 3), False),
    ((3, 3), True),
    ((3, 3, 3, 3), True),
])
def test_hog_incorrect_dimensions(shape, multichannel):
    img = np.zeros(shape)
    with pytest.raises(ValueError):
        with expected_warnings(["`multichannel` is a deprecated argument"]):
            feature.hog(img, multichannel=multichannel, block_norm='L1')


def test_hog_output_equivariance_deprecated_multichannel():
    img = data.astronaut()
    img[:, :, (1, 2)] = 0
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        hog_ref = feature.hog(img, multichannel=True, block_norm='L1')

    for n in (1, 2):
        with expected_warnings(["`multichannel` is a deprecated argument"]):
            hog_fact = feature.hog(np.roll(img, n, axis=2), multichannel=True,
                                   block_norm='L1')
        assert_almost_equal(hog_ref, hog_fact)

        # repeat prior test, but check for positional multichannel warning
        with expected_warnings(["Providing the `multichannel` argument"]):
            hog_fact = feature.hog(np.roll(img, n, axis=2), 9, (8, 8), (3, 3),
                                   'L1', False, False, True, True)
        assert_almost_equal(hog_ref, hog_fact)


@pytest.mark.parametrize('channel_axis', [0, 1, -1, -2])
def test_hog_output_equivariance_channel_axis(channel_axis):
    img = data.astronaut()[:64, :32]
    img[:, :, (1, 2)] = 0
    img = np.moveaxis(img, -1, channel_axis)
    hog_ref = feature.hog(img, channel_axis=channel_axis, block_norm='L1')

    for n in (1, 2):
        hog_fact = feature.hog(np.roll(img, n, axis=channel_axis),
                               channel_axis=channel_axis, block_norm='L1')
        assert_almost_equal(hog_ref, hog_fact)
