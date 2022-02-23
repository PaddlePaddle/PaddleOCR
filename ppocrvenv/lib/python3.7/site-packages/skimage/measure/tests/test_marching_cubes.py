import numpy as np
import pytest
from numpy.testing import assert_allclose

from skimage.draw import ellipsoid, ellipsoid_stats
from skimage.measure import marching_cubes, mesh_surface_area


def test_marching_cubes_isotropic():
    ellipsoid_isotropic = ellipsoid(6, 10, 16, levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)

    # Classic
    verts, faces = marching_cubes(ellipsoid_isotropic, 0., method='_lorensen')
    surf_calc = mesh_surface_area(verts, faces)
    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99

    # Lewiner
    verts, faces = marching_cubes(ellipsoid_isotropic, 0.)[:2]
    surf_calc = mesh_surface_area(verts, faces)
    # Test within 1% tolerance for isotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.99


def test_marching_cubes_anisotropic():
    # test spacing as numpy array (and not just tuple)
    spacing = np.array([1., 10 / 6., 16 / 6.])
    ellipsoid_anisotropic = ellipsoid(6, 10, 16, spacing=spacing,
                                      levelset=True)
    _, surf = ellipsoid_stats(6, 10, 16)

    # Classic
    verts, faces = marching_cubes(ellipsoid_anisotropic, 0.,
                                  spacing=spacing, method='_lorensen')
    surf_calc = mesh_surface_area(verts, faces)
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    # Lewiner
    verts, faces = marching_cubes(ellipsoid_anisotropic, 0.,
                                  spacing=spacing)[:2]
    surf_calc = mesh_surface_area(verts, faces)
    # Test within 1.5% tolerance for anisotropic. Will always underestimate.
    assert surf > surf_calc and surf_calc > surf * 0.985

    # Test marching cube with mask
    with pytest.raises(ValueError):
        verts, faces = marching_cubes(
            ellipsoid_anisotropic, 0., spacing=spacing,
            mask=np.array([]))[:2]

    # Test spacing together with allow_degenerate=False
    marching_cubes(ellipsoid_anisotropic, 0, spacing=spacing,
                   allow_degenerate=False)


def test_invalid_input():
    # Classic
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 0, method='_lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 1, method='_lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.ones((3, 3, 3)), 1, spacing=(1, 2), method='_lorensen')
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((20, 20)), 0, method='_lorensen')

    # Lewiner
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 0)
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((2, 2, 1)), 1)
    with pytest.raises(ValueError):
        marching_cubes(np.ones((3, 3, 3)), 1, spacing=(1, 2))
    with pytest.raises(ValueError):
        marching_cubes(np.zeros((20, 20)), 0)


def test_both_algs_same_result_ellipse():
    # Performing this test on data that does not have ambiguities

    sphere_small = ellipsoid(1, 1, 1, levelset=True)

    vertices1, faces1 = marching_cubes(sphere_small, 0, method='_lorensen')[:2]
    vertices2, faces2 = marching_cubes(sphere_small, 0,
                                       allow_degenerate=False)[:2]
    vertices3, faces3 = marching_cubes(sphere_small, 0,
                                       allow_degenerate=False,
                                       method='lorensen')[:2]

    # Order is different, best we can do is test equal shape and same
    # vertices present
    assert _same_mesh(vertices1, faces1, vertices2, faces2)
    assert _same_mesh(vertices1, faces1, vertices3, faces3)


def _same_mesh(vertices1, faces1, vertices2, faces2, tol=1e-10):
    """ Compare two meshes, using a certain tolerance and invariant to
    the order of the faces.
    """
    # Unwind vertices
    triangles1 = vertices1[np.array(faces1)]
    triangles2 = vertices2[np.array(faces2)]
    # Sort vertices within each triangle
    triang1 = [np.concatenate(sorted(t, key=lambda x:tuple(x)))
               for t in triangles1]
    triang2 = [np.concatenate(sorted(t, key=lambda x:tuple(x)))
               for t in triangles2]
    # Sort the resulting 9-element "tuples"
    triang1 = np.array(sorted([tuple(x) for x in triang1]))
    triang2 = np.array(sorted([tuple(x) for x in triang2]))
    return (triang1.shape == triang2.shape and
            np.allclose(triang1, triang2, 0, tol))


def test_both_algs_same_result_donut():
    # Performing this test on data that does not have ambiguities
    n = 48
    a, b = 2.5/n, -1.25

    vol = np.empty((n, n, n), 'float32')
    for iz in range(vol.shape[0]):
        for iy in range(vol.shape[1]):
            for ix in range(vol.shape[2]):
                # Double-torii formula by Thomas Lewiner
                z, y, x = float(iz)*a+b, float(iy)*a+b, float(ix)*a+b
                vol[iz,iy,ix] = ( (
                    (8*x)**2 + (8*y-2)**2 + (8*z)**2 + 16 - 1.85*1.85 ) * ( (8*x)**2 +
                    (8*y-2)**2 + (8*z)**2 + 16 - 1.85*1.85 ) - 64 * ( (8*x)**2 + (8*y-2)**2 )
                    ) * ( ( (8*x)**2 + ((8*y-2)+4)*((8*y-2)+4) + (8*z)**2 + 16 - 1.85*1.85 )
                    * ( (8*x)**2 + ((8*y-2)+4)*((8*y-2)+4) + (8*z)**2 + 16 - 1.85*1.85 ) -
                    64 * ( ((8*y-2)+4)*((8*y-2)+4) + (8*z)**2
                    ) ) + 1025

    vertices1, faces1 = marching_cubes(vol, 0, method='_lorensen')[:2]
    vertices2, faces2 = marching_cubes(vol, 0)[:2]
    vertices3, faces3 = marching_cubes(vol, 0, method='lorensen')[:2]

    # Old and new alg are different
    assert not _same_mesh(vertices1, faces1, vertices2, faces2)
    # New classic and new Lewiner are different
    assert not _same_mesh(vertices2, faces2, vertices3, faces3)
    # Would have been nice if old and new classic would have been the same
    # assert _same_mesh(vertices1, faces1, vertices3, faces3, 5)


def test_masked_marching_cubes():

    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.ones_like(ellipsoid_scalar, dtype=bool)
    mask[:10, :, :] = False
    mask[:, :, 20:] = False
    ver, faces, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    area = mesh_surface_area(ver, faces)

    assert_allclose(area, 299.56878662109375, rtol=.01)


def test_masked_marching_cubes_empty():
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.array([])
    with pytest.raises(ValueError):
        _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)


def test_masked_marching_cubes_old_lewiner():
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.array([])
    with pytest.raises(NotImplementedError):
        _ = marching_cubes(ellipsoid_scalar, 0, mask=mask, method='_lorensen')


def test_masked_marching_cubes_all_true():
    ellipsoid_scalar = ellipsoid(6, 10, 16, levelset=True)
    mask = np.ones_like(ellipsoid_scalar, dtype=bool)
    ver_m, faces_m, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    ver, faces, _, _ = marching_cubes(ellipsoid_scalar, 0, mask=mask)
    assert_allclose(ver_m, ver, rtol=.00001)
    assert_allclose(faces_m, faces, rtol=.00001)
