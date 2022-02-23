import numpy as np
import skimage.data as data
from skimage.data._fetchers import image_fetcher
from skimage import io
from skimage._shared.testing import assert_equal, assert_almost_equal, fetch
import os
import pytest


def test_data_dir():
    # data_dir should be a directory people can use as a standard directory
    # https://github.com/scikit-image/scikit-image/pull/3945#issuecomment-498141893
    data_dir = data.data_dir
    assert 'astronaut.png' in os.listdir(data_dir)


def test_download_all_with_pooch():
    # jni first wrote this test with the intention of
    # fully deleting the files in the data_dir,
    # then ensure that the data gets downloaded accordingly.
    # hmaarrfk raised the concern that this test wouldn't
    # play well with parallel testing since we
    # may be breaking the global state that certain other
    # tests require, especially in parallel testing

    # The second concern is that this test essentially uses
    # alot of bandwidth, which is not fun for developers on
    # lower speed connections.
    # https://github.com/scikit-image/scikit-image/pull/4666/files/26d5138b25b958da6e97ebf979e9bc36f32c3568#r422604863
    data_dir = data.data_dir
    if image_fetcher is not None:
        data.download_all()
        assert len(os.listdir(data_dir)) > 50
    else:
        with pytest.raises(ModuleNotFoundError):
            data.download_all()


def test_astronaut():
    """ Test that "astronaut" image can be loaded. """
    astronaut = data.astronaut()
    assert_equal(astronaut.shape, (512, 512, 3))


def test_camera():
    """ Test that "camera" image can be loaded. """
    cameraman = data.camera()
    assert_equal(cameraman.ndim, 2)


def test_checkerboard():
    """ Test that "checkerboard" image can be loaded. """
    data.checkerboard()


def test_chelsea():
    """ Test that "chelsea" image can be loaded. """
    data.chelsea()


def test_clock():
    """ Test that "clock" image can be loaded. """
    data.clock()


def test_coffee():
    """ Test that "coffee" image can be loaded. """
    data.coffee()


def test_eagle():
    """ Test that "eagle" image can be loaded. """
    # Fetching the data through the testing module will
    # cause the test to skip if pooch isn't installed.
    fetch('data/eagle.png')
    eagle = data.eagle()
    assert_equal(eagle.ndim, 2)
    assert_equal(eagle.dtype, np.dtype('uint8'))


def test_horse():
    """ Test that "horse" image can be loaded. """
    horse = data.horse()
    assert_equal(horse.ndim, 2)
    assert_equal(horse.dtype, np.dtype('bool'))


def test_hubble():
    """ Test that "Hubble" image can be loaded. """
    data.hubble_deep_field()


def test_immunohistochemistry():
    """ Test that "immunohistochemistry" image can be loaded. """
    data.immunohistochemistry()


def test_logo():
    """ Test that "logo" image can be loaded. """
    logo = data.logo()
    assert_equal(logo.ndim, 3)
    assert_equal(logo.shape[2], 4)


def test_moon():
    """ Test that "moon" image can be loaded. """
    data.moon()


def test_page():
    """ Test that "page" image can be loaded. """
    data.page()


def test_rocket():
    """ Test that "rocket" image can be loaded. """
    data.rocket()


def test_text():
    """ Test that "text" image can be loaded. """
    data.text()


def test_stereo_motorcycle():
    """ Test that "stereo_motorcycle" image can be loaded. """
    data.stereo_motorcycle()


def test_binary_blobs():
    blobs = data.binary_blobs(length=128)
    assert_almost_equal(blobs.mean(), 0.5, decimal=1)
    blobs = data.binary_blobs(length=128, volume_fraction=0.25)
    assert_almost_equal(blobs.mean(), 0.25, decimal=1)
    blobs = data.binary_blobs(length=32, volume_fraction=0.25, n_dim=3)
    assert_almost_equal(blobs.mean(), 0.25, decimal=1)
    other_realization = data.binary_blobs(length=32, volume_fraction=0.25,
                                          n_dim=3)
    assert not np.all(blobs == other_realization)


def test_lfw_subset():
    """ Test that "lfw_subset" can be loaded."""
    data.lfw_subset()


def test_skin():
    """Test that "skin" image can be loaded.

    Needs internet connection.
    """
    skin = data.skin()
    assert skin.ndim == 3


def test_cell():
    """ Test that "cell" image can be loaded."""
    data.cell()


def test_cells3d():
    """Needs internet connection."""
    path = fetch('data/cells3d.tif')
    image = io.imread(path)
    assert image.shape == (60, 2, 256, 256)


def test_brain_3d():
    """Needs internet connection."""
    path = fetch('data/brain.tiff')
    image = io.imread(path)
    assert image.shape == (10, 256, 256)


def test_kidney_3d_multichannel():
    """Test that 3D multichannel image of kidney tissue can be loaded.

    Needs internet connection.
    """
    fetch('data/kidney.tif')
    kidney = data.kidney()
    assert kidney.shape == (16, 512, 512, 3)


def test_lily_multichannel():
    """Test that microscopy image of lily of the valley can be loaded.

    Needs internet connection.
    """
    fetch('data/lily.tif')
    lily = data.lily()
    assert lily.shape == (922, 922, 4)


def test_vortex():
    fetch('data/pivchallenge-B-B001_1.tif')
    fetch('data/pivchallenge-B-B001_2.tif')
    image0, image1 = data.vortex()
    for image in [image0, image1]:
        assert image.shape == (512, 512)


@pytest.mark.parametrize(
    'function_name', ['create_image_fetcher', 'file_hash', 'image_fetcher']
)
def test_fetchers_are_public(function_name):
    # Check that the following functions that are only used indirectly in the
    # above tests are public.
    assert hasattr(data, function_name)
