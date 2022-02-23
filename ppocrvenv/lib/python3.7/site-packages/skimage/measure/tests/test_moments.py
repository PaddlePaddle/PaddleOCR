import pytest
import numpy as np
from scipy import ndimage as ndi
from skimage import draw
from skimage.measure import (moments, moments_central, moments_coords,
                             moments_coords_central, moments_normalized,
                             moments_hu, centroid, inertia_tensor,
                             inertia_tensor_eigvals)

from skimage._shared import testing
from skimage._shared.testing import (assert_equal, assert_almost_equal,
                                     assert_allclose)
from skimage._shared.utils import _supported_float_type


def test_moments():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    m = moments(image)
    assert_equal(m[0, 0], 3)
    assert_almost_equal(m[1, 0] / m[0, 0], 14.5)
    assert_almost_equal(m[0, 1] / m[0, 0], 14.5)


def test_moments_central():
    image = np.zeros((20, 20), dtype=np.double)
    image[14, 14] = 1
    image[15, 15] = 1
    image[14, 15] = 0.5
    image[15, 14] = 0.5
    mu = moments_central(image, (14.5, 14.5))

    # check for proper centroid computation
    mu_calc_centroid = moments_central(image)
    assert_equal(mu, mu_calc_centroid)

    # shift image by dx=2, dy=2
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[16, 16] = 1
    image2[17, 17] = 1
    image2[16, 17] = 0.5
    image2[17, 16] = 0.5
    mu2 = moments_central(image2, (14.5 + 2, 14.5 + 2))
    # central moments must be translation invariant
    assert_equal(mu, mu2)


def test_moments_coords():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu_image = moments(image)

    coords = np.array([[r, c] for r in range(13, 17)
                       for c in range(13, 17)], dtype=np.double)
    mu_coords = moments_coords(coords)
    assert_almost_equal(mu_coords, mu_image)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_moments_coords_dtype(dtype):
    image = np.zeros((20, 20), dtype=dtype)
    image[13:17, 13:17] = 1

    expected_dtype = _supported_float_type(dtype)
    mu_image = moments(image)
    assert mu_image.dtype == expected_dtype

    coords = np.array([[r, c] for r in range(13, 17)
                       for c in range(13, 17)], dtype=dtype)
    mu_coords = moments_coords(coords)
    assert mu_coords.dtype == expected_dtype

    assert_almost_equal(mu_coords, mu_image)


def test_moments_central_coords():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu_image = moments_central(image, (14.5, 14.5))

    coords = np.array([[r, c] for r in range(13, 17)
                       for c in range(13, 17)], dtype=np.double)
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert_almost_equal(mu_coords, mu_image)

    # ensure that center is being calculated normally
    mu_coords_calc_centroid = moments_coords_central(coords)
    assert_almost_equal(mu_coords_calc_centroid, mu_coords)

    # shift image by dx=3 dy=3
    image = np.zeros((20, 20), dtype=np.double)
    image[16:20, 16:20] = 1
    mu_image = moments_central(image, (14.5, 14.5))

    coords = np.array([[r, c] for r in range(16, 20)
                       for c in range(16, 20)], dtype=np.double)
    mu_coords = moments_coords_central(coords, (14.5, 14.5))
    assert_almost_equal(mu_coords, mu_image)


def test_moments_normalized():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:17, 13:17] = 1
    mu = moments_central(image, (14.5, 14.5))
    nu = moments_normalized(mu)
    # shift image by dx=-3, dy=-3 and scale by 0.5
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[11:13, 11:13] = 1
    mu2 = moments_central(image2, (11.5, 11.5))
    nu2 = moments_normalized(mu2)
    # central moments must be translation and scale invariant
    assert_almost_equal(nu, nu2, decimal=1)


def test_moments_normalized_3d():
    image = draw.ellipsoid(1, 1, 10)
    mu_image = moments_central(image)
    nu = moments_normalized(mu_image)
    assert nu[0, 0, 2] > nu[0, 2, 0]
    assert_almost_equal(nu[0, 2, 0], nu[2, 0, 0])

    coords = np.where(image)
    mu_coords = moments_coords_central(coords)
    assert_almost_equal(mu_coords, mu_image)


def test_moments_normalized_invalid():
    with testing.raises(ValueError):
        moments_normalized(np.zeros((3, 3)), 3)
    with testing.raises(ValueError):
        moments_normalized(np.zeros((3, 3)), 4)


def test_moments_hu():
    image = np.zeros((20, 20), dtype=np.double)
    image[13:15, 13:17] = 1
    mu = moments_central(image, (13.5, 14.5))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    # shift image by dx=2, dy=3, scale by 0.5 and rotate by 90deg
    image2 = np.zeros((20, 20), dtype=np.double)
    image2[11, 11:13] = 1
    image2 = image2.T
    mu2 = moments_central(image2, (11.5, 11))
    nu2 = moments_normalized(mu2)
    hu2 = moments_hu(nu2)
    # central moments must be translation and scale invariant
    assert_almost_equal(hu, hu2, decimal=1)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_moments_dtype(dtype):
    image = np.zeros((20, 20), dtype=dtype)
    image[13:15, 13:17] = 1

    expected_dtype = _supported_float_type(dtype)
    mu = moments_central(image, (13.5, 14.5))
    assert mu.dtype == expected_dtype

    nu = moments_normalized(mu)
    assert nu.dtype == expected_dtype

    hu = moments_hu(nu)
    assert hu.dtype == expected_dtype


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_centroid(dtype):
    image = np.zeros((20, 20), dtype=dtype)
    image[14, 14:16] = 1
    image[15, 14:16] = 1/3
    image_centroid = centroid(image)
    if dtype == np.float16:
        rtol = 1e-3
    elif dtype == np.float32:
        rtol = 1e-5
    else:
        rtol = 1e-7
    assert_allclose(image_centroid, (14.25, 14.5), rtol=rtol)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_inertia_tensor_2d(dtype):
    image = np.zeros((40, 40), dtype=dtype)
    image[15:25, 5:35] = 1  # big horizontal rectangle (aligned with axis 1)
    expected_dtype = _supported_float_type(image.dtype)

    T = inertia_tensor(image)
    assert T.dtype == expected_dtype
    assert T[0, 0] > T[1, 1]
    np.testing.assert_allclose(T[0, 1], 0)

    v0, v1 = inertia_tensor_eigvals(image, T=T)
    assert v0.dtype == expected_dtype
    assert v1.dtype == expected_dtype
    np.testing.assert_allclose(np.sqrt(v0/v1), 3, rtol=0.01, atol=0.05)


def test_inertia_tensor_3d():
    image = draw.ellipsoid(10, 5, 3)
    T0 = inertia_tensor(image)
    eig0, V0 = np.linalg.eig(T0)
    # principal axis of ellipse = eigenvector of smallest eigenvalue
    v0 = V0[:, np.argmin(eig0)]

    assert np.allclose(v0, [1, 0, 0]) or np.allclose(-v0, [1, 0, 0])

    imrot = ndi.rotate(image.astype(float), 30, axes=(0, 1), order=1)
    Tr = inertia_tensor(imrot)
    eigr, Vr = np.linalg.eig(Tr)
    vr = Vr[:, np.argmin(eigr)]

    # Check that axis has rotated by expected amount
    pi, cos, sin = np.pi, np.cos, np.sin
    R = np.array([[ cos(pi/6), -sin(pi/6), 0],
                  [ sin(pi/6),  cos(pi/6), 0],
                  [         0,          0, 1]])
    expected_vr = R @ v0
    assert (np.allclose(vr, expected_vr, atol=1e-3, rtol=0.01) or
            np.allclose(-vr, expected_vr, atol=1e-3, rtol=0.01))


def test_inertia_tensor_eigvals():
    # Floating point precision problems could make a positive
    # semidefinite matrix have an eigenvalue that is very slightly
    # negative.  Check that we have caught and fixed this problem.
    image = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    # mu = np.array([[3, 0, 98], [0, 14, 0], [2, 0, 98]])
    eigvals = inertia_tensor_eigvals(image=image)
    assert (min(eigvals) >= 0)
