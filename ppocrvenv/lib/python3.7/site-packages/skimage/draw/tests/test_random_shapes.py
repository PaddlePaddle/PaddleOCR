import numpy as np
import pytest

from skimage.draw import random_shapes

from skimage._shared import testing
from skimage._shared._warnings import expected_warnings


def test_generates_color_images_with_correct_shape():
    image, _ = random_shapes((128, 128), max_shapes=10)
    assert image.shape == (128, 128, 3)


def test_generates_gray_images_with_correct_shape():
    image, _ = random_shapes(
        (4567, 123), min_shapes=3, max_shapes=20, channel_axis=None)
    assert image.shape == (4567, 123)


def test_generates_gray_images_with_correct_shape_deprecated_multichannel():
    with expected_warnings(["`multichannel` is a deprecated argument"]):
        image, _ = random_shapes(
            (4567, 123), min_shapes=3, max_shapes=20, multichannel=False)
    assert image.shape == (4567, 123)

    # repeat prior test, but check for positional multichannel warning
    with expected_warnings(["Providing the `multichannel` argument"]):
        image, _ = random_shapes((4567, 123), 20, 3, 2, None, False)
    assert image.shape == (4567, 123)


@pytest.mark.parametrize('channel_axis', [None, 0, 1, 2])
def test_generated_shape_for_channel_axis(channel_axis):
    shape = (128, 64)
    num_channels = 5

    image, _ = random_shapes(
        shape, num_channels=num_channels, min_shapes=3, max_shapes=10,
        channel_axis=channel_axis)

    if channel_axis is None:
        expected_shape = shape
    else:
        expected_shape = tuple(np.insert(shape, channel_axis, num_channels))

    assert image.shape == expected_shape


def test_generates_correct_bounding_boxes_for_rectangles():
    image, labels = random_shapes(
        (128, 128),
        max_shapes=1,
        shape='rectangle',
        random_seed=42)
    assert len(labels) == 1
    label, bbox = labels[0]
    assert label == 'rectangle', label

    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

    # The crop is filled.
    assert (crop >= 0).all() and (crop < 255).all()

    # The crop is complete.
    image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_triangles():
    image, labels = random_shapes(
        (128, 128),
        max_shapes=1,
        shape='triangle',
        random_seed=42)
    assert len(labels) == 1
    label, bbox = labels[0]
    assert label == 'triangle', label

    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

    # The crop is filled.
    assert (crop >= 0).any() and (crop < 255).any()

    # The crop is complete.
    image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_circles():
    image, labels = random_shapes(
        (43, 44),
        max_shapes=1,
        min_size=20,
        max_size=20,
        shape='circle',
        random_seed=42)
    assert len(labels) == 1
    label, bbox = labels[0]
    assert label == 'circle', label

    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

    # The crop is filled.
    assert (crop >= 0).any() and (crop < 255).any()

    # The crop is complete.
    image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = 255
    assert (image == 255).all()


def test_generates_correct_bounding_boxes_for_ellipses():
    image, labels = random_shapes(
        (43, 44),
        max_shapes=1,
        min_size=20,
        max_size=20,
        shape='ellipse',
        random_seed=42)
    assert len(labels) == 1
    label, bbox = labels[0]
    assert label == 'ellipse', label

    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

    # The crop is filled.
    assert (crop >= 0).any() and (crop < 255).any()

    # The crop is complete.
    image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]] = 255
    assert (image == 255).all()


def test_generate_circle_throws_when_size_too_small():
    with testing.raises(ValueError):
        random_shapes(
            (64, 128), max_shapes=1, min_size=1, max_size=1, shape='circle')


def test_generate_ellipse_throws_when_size_too_small():
    with testing.raises(ValueError):
        random_shapes(
            (64, 128), max_shapes=1, min_size=1, max_size=1, shape='ellipse')


def test_generate_triangle_throws_when_size_too_small():
    with testing.raises(ValueError):
        random_shapes(
            (128, 64), max_shapes=1, min_size=1, max_size=1, shape='triangle')


def test_can_generate_one_by_one_rectangle():
    image, labels = random_shapes(
        (50, 128),
        max_shapes=1,
        min_size=1,
        max_size=1,
        shape='rectangle')
    assert len(labels) == 1
    _, bbox = labels[0]
    crop = image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

    # rgb
    assert (np.shape(crop) == (1, 1, 3) and np.any(crop >= 1)
            and np.any(crop < 255))


def test_throws_when_intensity_range_out_of_range():
    with testing.raises(ValueError):
        with expected_warnings(["`multichannel` is a deprecated argument"]):
            random_shapes((1000, 1234), max_shapes=1, multichannel=False,
                          intensity_range=(0, 256))
    with testing.raises(ValueError):
        random_shapes((2, 2), max_shapes=1,
                      intensity_range=((-1, 255),))


def test_returns_empty_labels_and_white_image_when_cannot_fit_shape():
    # The circle will never fit this.
    with expected_warnings(['Could not fit']):
        image, labels = random_shapes(
            (10000, 10000), max_shapes=1, min_size=10000, shape='circle')
    assert len(labels) == 0
    assert (image == 255).all()


def test_random_shapes_is_reproducible_with_seed():
    random_seed = 42
    labels = []
    for _ in range(5):
        _, label = random_shapes((128, 128), max_shapes=5,
                                 random_seed=random_seed)
        labels.append(label)
    assert all(other == labels[0] for other in labels[1:])


def test_generates_white_image_when_intensity_range_255():
    image, labels = random_shapes((128, 128), max_shapes=3,
                                  intensity_range=((255, 255),),
                                  random_seed=42)
    assert len(labels) > 0
    assert (image == 255).all()
