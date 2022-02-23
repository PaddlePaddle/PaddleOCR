import numpy as np
from numpy.testing import (assert_equal, assert_allclose,
                           assert_almost_equal)
import pytest

pytest.importorskip("matplotlib")

from skimage import util
import skimage.data as data
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.viewer import ImageViewer, has_qt
from skimage.viewer.plugins.base import Plugin
from skimage.viewer.widgets import Slider
from skimage.viewer.plugins import (
    LineProfile, Measure, CannyPlugin, LabelPainter, Crop, ColorHistogram,
    PlotPlugin)


def setup_line_profile(image, limits='image'):
    viewer = ImageViewer(util.img_as_float(image))
    plugin = LineProfile(limits=limits)
    viewer += plugin
    return plugin


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_line_profile():
    """ Test a line profile using an ndim=2 image"""
    plugin = setup_line_profile(data.camera())
    line_image, scan_data = plugin.output()
    for inp in [line_image.nonzero()[0].size,
                line_image.sum() / line_image.max(),
                scan_data.size]:
        assert_equal(inp, 172)
    assert_equal(line_image.shape, (512, 512))
    assert_allclose(scan_data.max(), 0.886275, rtol=1e-3)
    assert_allclose(scan_data.mean(), 0.247834, rtol=1e-3)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_line_profile_rgb():
    """ Test a line profile using an ndim=3 image"""
    plugin = setup_line_profile(data.chelsea(), limits=None)
    for i in range(6):
        plugin.line_tool._thicken_scan_line()
    line_image, scan_data = plugin.output()
    assert_equal(line_image[line_image == 128].size, 906)
    assert_equal(line_image[line_image == 255].size, 151)
    assert_equal(line_image.shape, (300, 451))
    assert_equal(scan_data.shape, (151, 3))
    assert_allclose(scan_data.max(), 0.772, rtol=1e-3)
    assert_allclose(scan_data.mean(), 0.4359, rtol=1e-3)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_line_profile_dynamic():
    """Test a line profile updating after an image transform"""
    image = data.coins()[:-50, :]  # shave some off to make the line lower
    image = util.img_as_float(image)
    viewer = ImageViewer(image)

    lp = LineProfile(limits='dtype')
    viewer += lp

    line = lp.get_profiles()[-1][0]
    assert line.size == 129
    assert_almost_equal(np.std(viewer.image), 0.208, 3)
    assert_almost_equal(np.std(line), 0.229, 3)
    assert_almost_equal(np.max(line) - np.min(line), 0.725, 1)

    viewer.image = util.img_as_float(
        median(util.img_as_ubyte(image), footprint=disk(radius=3)))

    line = lp.get_profiles()[-1][0]
    assert_almost_equal(np.std(viewer.image), 0.198, 3)
    assert_almost_equal(np.std(line), 0.220, 3)
    assert_almost_equal(np.max(line) - np.min(line), 0.639, 1)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_measure():
    image = data.camera()
    viewer = ImageViewer(image)
    m = Measure()
    viewer += m

    m.line_changed([(0, 0), (10, 10)])
    assert_equal(str(m._length.text), '14.1')
    assert_equal(str(m._angle.text[:5]), '135.0')


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_canny():
    image = data.camera()
    viewer = ImageViewer(image)
    c = CannyPlugin()
    viewer += c

    canny_edges = viewer.show(False)
    viewer.close()
    edges = canny_edges[0][0]
    assert edges.sum() == 3590


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_label_painter():
    image = data.camera()
    moon = data.moon()
    viewer = ImageViewer(image)
    lp = LabelPainter()
    viewer += lp

    assert_equal(lp.radius, 5)
    lp.label = 1
    assert_equal(str(lp.label), '1')
    lp.label = 2
    assert_equal(str(lp.paint_tool.label), '2')
    assert_equal(lp.paint_tool.radius, 5)
    lp._on_new_image(moon)
    assert_equal(lp.paint_tool.shape, moon.shape)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_crop():
    image = data.camera()
    viewer = ImageViewer(image)
    c = Crop()
    viewer += c

    c.crop((0, 100, 0, 100))
    assert_equal(viewer.image.shape, (101, 101))


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_color_histogram():
    image = util.img_as_float(data.colorwheel())
    viewer = ImageViewer(image)
    ch = ColorHistogram(dock='right')
    viewer += ch

    assert_almost_equal(viewer.image.std(), 0.352, 3),
    ch.ab_selected((0, 100, 0, 100)),
    assert_almost_equal(viewer.image.std(), 0.325, 3)


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_plot_plugin():
    viewer = ImageViewer(data.moon())
    plugin = PlotPlugin(image_filter=lambda x: x)
    viewer += plugin

    assert_equal(viewer.image, data.moon())
    plugin._update_original_image(data.coins())
    assert_equal(viewer.image, data.coins())
    viewer.close()


@pytest.mark.skipif(not has_qt, reason="Qt not installed")
def test_plugin():
    img = util.img_as_float(data.moon())
    viewer = ImageViewer(img)

    def median_filter(img, radius=3):
        return median(
            util.img_as_ubyte(img), footprint=disk(radius=radius))

    plugin = Plugin(image_filter=median_filter)
    viewer += plugin

    plugin += Slider('radius', 1, 5)

    assert_almost_equal(np.std(viewer.image), 12.556, 3)

    plugin.filter_image()

    assert_almost_equal(np.std(viewer.image), 12.931, 3)

    plugin.show()
    plugin.close()
    plugin.clean_up()
    img, _ = plugin.output()
    assert_equal(img, viewer.image)
