import numpy as np
import pytest
from scipy.signal import get_window

from skimage.filters import window


@pytest.mark.parametrize("size", [5, 6])
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_window_shape_isotropic(size, ndim):
    w = window('hann', (size,)*ndim)
    assert w.ndim == ndim
    assert w.shape[1:] == w.shape[:-1]
    for i in range(1, ndim-1):
        assert np.allclose(w.sum(axis=0), w.sum(axis=i))


@pytest.mark.parametrize("shape", [(8, 16), (16, 8), (2, 3, 4)])
def test_window_shape_anisotropic(shape):
    w = window('hann', shape)
    assert w.shape == shape


@pytest.mark.parametrize("shape", [[17, 33], [17, 97]])
def test_window_anisotropic_amplitude(shape):
    w = window(('tukey', 0.8), shape)

    # The shape is stretched to give approximately the same range on each axis,
    # so the center profile should have a similar mean value.
    profile_w = w[w.shape[0]//2, :]
    profile_h = w[:, w.shape[1]//2]
    assert abs(profile_w.mean() - profile_h.mean()) < .01


@pytest.mark.parametrize(
    "wintype", [16, 'triang', ('tukey', 0.8)]
)
def test_window_type(wintype):
    w = window(wintype, (9, 9))
    assert w.ndim == 2
    assert w.shape[1:] == w.shape[:-1]
    assert np.allclose(w.sum(axis=0), w.sum(axis=1))


@pytest.mark.parametrize("size", [10, 11])
def test_window_1d(size):
    w = window('hann', size)
    w1 = get_window('hann', size, fftbins=False)
    assert np.allclose(w, w1)


def test_window_invalid_shape():
    with pytest.raises(ValueError):
        window(10, shape=(-5, 10))
    with pytest.raises(ValueError):
        window(10, shape=(1.3, 2.0))
