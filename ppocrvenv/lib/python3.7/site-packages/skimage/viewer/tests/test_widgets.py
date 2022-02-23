import os

from numpy.testing import assert_almost_equal, assert_equal
import pytest

pytest.importorskip("matplotlib")

from skimage import data, img_as_float, io, img_as_ubyte

from skimage.viewer import ImageViewer
from skimage.viewer.qt import QtWidgets, QtCore, has_qt
from skimage.viewer.widgets import (
    Slider, OKCancelButtons, SaveButtons, ComboBox, CheckBox, Text)
from skimage.viewer.plugins.base import Plugin


def get_image_viewer():
    image = data.coins()
    viewer = ImageViewer(img_as_float(image))
    viewer += Plugin()
    return viewer


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_check_box():
    viewer = get_image_viewer()
    cb = CheckBox('hello', value=True, alignment='left')
    viewer.plugins[0] += cb

    assert_equal(cb.val, True)
    cb.val = False
    assert_equal(cb.val, False)
    cb.val = 1
    assert_equal(cb.val, True)
    cb.val = 0
    assert_equal(cb.val, False)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_combo_box():
    viewer = get_image_viewer()
    cb = ComboBox('hello', ('a', 'b', 'c'))
    viewer.plugins[0] += cb

    assert_equal(str(cb.val), 'a')
    assert_equal(cb.index, 0)
    cb.index = 2
    assert_equal(str(cb.val), 'c'),
    assert_equal(cb.index, 2)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_text_widget():
    viewer = get_image_viewer()
    txt = Text('hello', 'hello, world!')
    viewer.plugins[0] += txt

    assert_equal(str(txt.text), 'hello, world!')
    txt.text = 'goodbye, world!'
    assert_equal(str(txt.text), 'goodbye, world!')


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_slider_int():
    viewer = get_image_viewer()
    sld = Slider('radius', 2, 10, value_type='int')
    viewer.plugins[0] += sld

    assert_equal(sld.val, 4)
    sld.val = 6
    assert_equal(sld.val, 6)
    sld.editbox.setText('5')
    sld._on_editbox_changed()
    assert_equal(sld.val, 5)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_slider_float():
    viewer = get_image_viewer()
    sld = Slider('alpha', 2.1, 3.1, value=2.1, value_type='float',
                 orientation='vertical', update_on='move')
    viewer.plugins[0] += sld

    assert_equal(sld.val, 2.1)
    sld.val = 2.5
    assert_almost_equal(sld.val, 2.5, 2)
    sld.editbox.setText('0.1')
    sld._on_editbox_changed()
    assert_almost_equal(sld.val, 2.5, 2)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_save_buttons():
    viewer = get_image_viewer()
    sv = SaveButtons()
    viewer.plugins[0] += sv

    import tempfile
    fid, filename = tempfile.mkstemp(suffix='.png')
    os.close(fid)

    timer = QtCore.QTimer()
    timer.singleShot(100, QtWidgets.QApplication.quit)

    # exercise the button clicks
    sv.save_stack.click()
    sv.save_file.click()

    # call the save functions directly
    sv.save_to_stack()
    sv.save_to_file(filename)

    img = io.imread(filename)

    assert_almost_equal(img, img_as_ubyte(viewer.image))

    img = io.pop()
    assert_almost_equal(img, viewer.image)

    os.remove(filename)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_ok_buttons():
    viewer = get_image_viewer()
    ok = OKCancelButtons()
    viewer.plugins[0] += ok

    ok.update_original_image(),
    ok.close_plugin()
