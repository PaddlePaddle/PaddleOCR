import pytest
import numpy as np
from skimage import data
from skimage.measure._label import _label_bool, label
from skimage.measure._ccomp import label_cython as clabel

from skimage._shared import testing

# In this testsuite, we ensure that the results provided by
# label_cython are identical to the one from _label_bool,
# which is based on ndimage.


def test_no_option():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img)
    l_cy = clabel(img)
    testing.assert_equal(l_ndi, l_cy)


def test_background():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img, background=0)
    l_cy = clabel(img, background=0)
    testing.assert_equal(l_ndi, l_cy)

    l_ndi = _label_bool(img, background=1)
    l_cy = clabel(img, background=1)
    testing.assert_equal(l_ndi, l_cy)


def test_return_num():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    l_ndi = _label_bool(img, return_num=True)
    l_cy = clabel(img, return_num=True)
    testing.assert_equal(l_ndi, l_cy)


def test_connectivity():
    img = data.binary_blobs(length=128, blob_size_fraction=0.15, n_dim=3)
    for c in (1, 2, 3):
        l_ndi = _label_bool(img, connectivity=c)
        l_cy = clabel(img, connectivity=c)
        testing.assert_equal(l_ndi, l_cy)

    for c in (0, 4):
        with pytest.raises(ValueError):
            l_ndi = _label_bool(img, connectivity=c)
        with pytest.raises(ValueError):
            l_cy = clabel(img, connectivity=c)


@pytest.mark.parametrize("dtype", [bool, int])
def test_zero_size(dtype):
    img = np.ones((300, 0, 300), dtype=dtype)
    lab, num = label(img, return_num=True)

    assert lab.shape == img.shape
    assert num == 0
