from numpy.testing import assert_equal
import pytest

pytest.importorskip("matplotlib")

from skimage._shared.version_requirements import is_installed
from skimage import data
from skimage.filters import sobel
from skimage.transform import pyramid_gaussian

from skimage.viewer.qt import QtGui, QtCore, has_qt
from skimage.viewer import ImageViewer, CollectionViewer
from skimage.viewer.plugins import OverlayPlugin


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_viewer():
    astro = data.astronaut()
    coins = data.coins()

    view = ImageViewer(astro)
    import tempfile
    _, filename = tempfile.mkstemp(suffix='.png')

    view.show(False)
    view.close()
    view.save_to_file(filename)
    view.open_file(filename)
    assert_equal(view.image, astro)
    view.image = coins
    assert_equal(view.image, coins),
    view.save_to_file(filename),
    view.open_file(filename),
    view.reset_image(),
    assert_equal(view.image, coins)


def make_key_event(key):
    return QtGui.QKeyEvent(QtCore.QEvent.KeyPress, key,
                           QtCore.Qt.NoModifier)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_collection_viewer():

    img = data.astronaut()
    img_collection = tuple(pyramid_gaussian(img, channel_axis=-1))

    view = CollectionViewer(img_collection)
    make_key_event(48)

    view.update_index('', 2),
    assert_equal(view.image, img_collection[2])
    view.keyPressEvent(make_key_event(53))
    assert_equal(view.image, img_collection[5])
    view._format_coord(10, 10)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
@pytest.mark.skipif(not is_installed('matplotlib', '>=1.2'),
                reason="matplotlib < 1.2")
def test_viewer_with_overlay():
    img = data.coins()
    ov = OverlayPlugin(image_filter=sobel)
    viewer = ImageViewer(img)
    viewer += ov

    import tempfile
    _, filename = tempfile.mkstemp(suffix='.png')

    ov.color = 3
    assert_equal(ov.color, 'yellow')

    viewer.save_to_file(filename)
    ov.display_filtered_image(img)
    assert_equal(ov.overlay, img)
    ov.overlay = None
    assert_equal(ov.overlay, None)
    ov.overlay = img
    assert_equal(ov.overlay, img)
    assert_equal(ov.filtered_image, img)
