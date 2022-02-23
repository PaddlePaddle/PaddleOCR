import os

import numpy as np
from skimage.io import use_plugin, reset_plugins
from skimage.io.collection import MultiImage

from skimage._shared import testing
from skimage._shared.testing import assert_equal, assert_allclose

from pytest import fixture

@fixture
def imgs():
    use_plugin('pil')

    paths = [testing.fetch('data/multipage_rgb.tif'),
             testing.fetch('data/no_time_for_that_tiny.gif')]
    imgs = [MultiImage(paths[0]),
            MultiImage(paths[0], conserve_memory=False),
            MultiImage(paths[1]),
            MultiImage(paths[1], conserve_memory=False),
            MultiImage(os.pathsep.join(paths))]
    yield imgs

    reset_plugins()

def test_shapes(imgs):
    imgs = imgs[-1]
    assert imgs[0][0].shape == imgs[0][1].shape
    assert imgs[0][0].shape == (10, 10, 3)

def test_len(imgs):
    assert len(imgs[0][0]) == len(imgs[1][0]) == 2
    assert len(imgs[2][0]) == len(imgs[3][0]) == 24
    assert len(imgs[-1]) == 2, len(imgs[-1])

def test_slicing(imgs):
    img = imgs[-1]
    assert type(img[:]) is MultiImage
    assert len(img[0][:]) + len(img[1][:]) == 26, len(img[:])
    assert len(img[0][:1]) == 1
    assert len(img[1][1:]) == 23
    assert_allclose(img[0], img[:1][0])
    assert_allclose(img[1], img[1:][0])
    assert_allclose(img[-1], img[::-1][0])
    assert_allclose(img[0], img[::-1][-1])

def test_getitem(imgs):
    for img in imgs[0]:
        num = len(img)

        for i in range(-num, num):
            assert type(img[i]) is np.ndarray
        assert_allclose(img[0], img[-num])

        with testing.raises(AssertionError):
            assert_allclose(img[0], img[1])

        with testing.raises(IndexError):
            img[num]
        with testing.raises(IndexError):
            img[-num - 1]

def test_files_property(imgs):
    for img in imgs:
        if isinstance(img, MultiImage):
            continue

        assert isinstance(img.filename, str)

        with testing.raises(AttributeError):
            img.filename = "newfile"

def test_conserve_memory_property(imgs):
    for img in imgs:
        assert isinstance(img.conserve_memory, bool)

        with testing.raises(AttributeError):
            img.conserve_memory = True

def test_concatenate(imgs):
    for img in imgs:
        if img[0].shape != img[-1].shape:
            with testing.raises(ValueError):
                img.concatenate()
            continue
        array = img.concatenate()
        assert_equal(array.shape, (len(img),) + img[0].shape)
