import numpy as np
import skimage.io as io
from skimage._shared import testing

testing.pytest.importorskip('astropy')
from astropy.io import fits
import skimage.io._plugins.fits_plugin as fplug


def test_fits_plugin_import():
    # Make sure we get an import exception if Astropy isn't there
    # (not sure how useful this is, but it ensures there isn't some other
    # error when trying to load the plugin)
    try:
        io.use_plugin('fits')
    except ImportError:
        raise()


def teardown():
    io.reset_plugins()


def _same_ImageCollection(collection1, collection2):
    """
    Ancillary function to compare two ImageCollection objects, checking that
    their constituent arrays are equal.
    """
    if len(collection1) != len(collection2):
        return False
    for ext1, ext2 in zip(collection1, collection2):
        if not np.all(ext1 == ext2):
            return False
    return True
