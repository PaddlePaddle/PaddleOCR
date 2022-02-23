import numpy as np
import pytest

from skimage import io
from skimage._shared._warnings import expected_warnings

plt = pytest.importorskip("matplotlib.pyplot")


def setup():
    io.reset_plugins()

# test images. Note that they don't have their full range for their dtype,
# but we still expect the display range to equal the full dtype range.
im8 = np.array([[0, 64], [128, 240]], np.uint8)
im16 = im8.astype(np.uint16) * 256
im64 = im8.astype(np.uint64)
imf = im8 / 255
im_lo = imf / 1000
im_hi = imf + 10

imshow_expected_warnings = [
    r"tight_layout : falling back to Agg|\A\Z",
    r"tight_layout: falling back to Agg|\A\Z",  # formatting change in mpl
    # Maptlotlib 2.2.3 seems to use np.asscalar which issues a warning
    # with numpy 1.16
    # Matplotlib 2.2.3 is the last supported version for python 2.7
    r"np.asscalar|\A\Z"
]


def n_subplots(ax_im):
    """Return the number of subplots in the figure containing an ``AxesImage``.

    Parameters
    ----------
    ax_im : matplotlib.pyplot.AxesImage object
        The input ``AxesImage``.

    Returns
    -------
    n : int
        The number of subplots in the corresponding figure.

    Notes
    -----
    This function is intended to check whether a colorbar was drawn, in
    which case two subplots are expected. For standard imshows, one
    subplot is expected.
    """
    return len(ax_im.get_figure().get_axes())


def test_uint8():
    plt.figure()
    with expected_warnings(imshow_expected_warnings +
                           [r"CObject type is marked|\A\Z"]):
        ax_im = io.imshow(im8)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 255)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None


def test_uint16():
    plt.figure()
    with expected_warnings(imshow_expected_warnings +
                           [r"CObject type is marked|\A\Z"]):
        ax_im = io.imshow(im16)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 65535)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None


def test_float():
    plt.figure()
    with expected_warnings(imshow_expected_warnings +
                           [r"CObject type is marked|\A\Z"]):
        ax_im = io.imshow(imf)
    assert ax_im.cmap.name == 'gray'
    assert ax_im.get_clim() == (0, 1)
    assert n_subplots(ax_im) == 1
    assert ax_im.colorbar is None


def test_low_data_range():
    with expected_warnings(imshow_expected_warnings +
                           ["Low image data range|CObject type is marked"]):
        ax_im = io.imshow(im_lo)
    assert ax_im.get_clim() == (im_lo.min(), im_lo.max())
    # check that a colorbar was created
    assert ax_im.colorbar is not None


def test_outside_standard_range():
    plt.figure()
    # Warning raised by matplotlib on Windows:
    # "The CObject type is marked Pending Deprecation in Python 2.7.
    #  Please use capsule objects instead."
    # Ref: https://docs.python.org/2/c-api/cobject.html
    with expected_warnings(imshow_expected_warnings +
                           ["out of standard range|CObject type is marked"]):
        ax_im = io.imshow(im_hi)
    assert ax_im.get_clim() == (im_hi.min(), im_hi.max())
    assert n_subplots(ax_im) == 2
    assert ax_im.colorbar is not None


def test_nonstandard_type():
    plt.figure()
    # Warning raised by matplotlib on Windows:
    # "The CObject type is marked Pending Deprecation in Python 2.7.
    #  Please use capsule objects instead."
    # Ref: https://docs.python.org/2/c-api/cobject.html
    with expected_warnings(imshow_expected_warnings +
                           ["Low image data range|CObject type is marked"]):
        ax_im = io.imshow(im64)
    assert ax_im.get_clim() == (im64.min(), im64.max())
    assert n_subplots(ax_im) == 2
    assert ax_im.colorbar is not None


def test_signed_image():
    plt.figure()
    im_signed = np.array([[-0.5, -0.2], [0.1, 0.4]])

    with expected_warnings(imshow_expected_warnings +
                           [r"CObject type is marked|\A\Z"]):
        ax_im = io.imshow(im_signed)
    assert ax_im.get_clim() == (-0.5, 0.5)
    assert n_subplots(ax_im) == 2
    assert ax_im.colorbar is not None
