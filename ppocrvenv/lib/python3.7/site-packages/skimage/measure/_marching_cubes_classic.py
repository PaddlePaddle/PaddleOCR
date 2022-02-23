import numpy as np
import scipy.ndimage as ndi
from . import _marching_cubes_classic_cy


def _marching_cubes_classic(volume, level, spacing, gradient_direction):
    """Lorensen et al. algorithm for marching cubes. See
    marching_cubes_classic for documentation.

    """
    # Check inputs and ensure `volume` is C-contiguous for memoryviews
    if volume.ndim != 3:
        raise ValueError("Input volume must have 3 dimensions.")
    if level is None:
        level = 0.5 * (volume.min() + volume.max())
    else:
        level = float(level)
        if level < volume.min() or level > volume.max():
            raise ValueError("Surface level must be within volume data range.")
    if len(spacing) != 3:
        raise ValueError("`spacing` must consist of three floats.")

    volume = np.array(volume, dtype=np.float64, order="C")

    # Extract raw triangles using marching cubes in Cython
    #   Returns a list of length-3 lists, each sub-list containing three
    #   tuples. The tuples hold (x, y, z) coordinates for triangle vertices.
    # Note: this algorithm is fast, but returns degenerate "triangles" which
    #   have repeated vertices - and equivalent vertices are redundantly
    #   placed in every triangle they connect with.
    raw_faces = _marching_cubes_classic_cy.iterate_and_store_3d(volume,
                                                                float(level))

    # Find and collect unique vertices, storing triangle verts as indices.
    # Returns a true mesh with no degenerate faces.
    verts, faces = _marching_cubes_classic_cy.unpack_unique_verts(raw_faces)

    verts = np.asarray(verts)
    faces = np.asarray(faces)

    # Fancy indexing to define two vector arrays from triangle vertices
    faces = _correct_mesh_orientation(volume, verts[faces], faces, spacing,
                                      gradient_direction)

    # Adjust for non-isotropic spacing in `verts` at time of return
    return verts * np.r_[spacing], faces


def mesh_surface_area(verts, faces):
    """
    Compute surface area, given vertices & triangular faces

    Parameters
    ----------
    verts : (V, 3) array of floats
        Array containing (x, y, z) coordinates for V unique mesh vertices.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in `verts`

    Returns
    -------
    area : float
        Surface area of mesh. Units now [coordinate units] ** 2.

    Notes
    -----
    The arguments expected by this function are the first two outputs from
    `skimage.measure.marching_cubes`. For unit correct output, ensure correct
    `spacing` was passed to `skimage.measure.marching_cubes`.

    This algorithm works properly only if the ``faces`` provided are all
    triangles.

    See Also
    --------
    skimage.measure.marching_cubes
    skimage.measure.marching_cubes_classic

    """
    # Fancy indexing to define two vector arrays from triangle vertices
    actual_verts = verts[faces]
    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]
    del actual_verts

    # Area of triangle in 3D = 1/2 * Euclidean norm of cross product
    return ((np.cross(a, b) ** 2).sum(axis=1) ** 0.5).sum() / 2.


def _correct_mesh_orientation(volume, actual_verts, faces,
                              spacing=(1., 1., 1.),
                              gradient_direction='descent'):
    """
    Correct orientations of mesh faces.

    Parameters
    ----------
    volume : (M, N, P) array of doubles
        Input data volume to find isosurfaces. Will be cast to `np.float64`.
    actual_verts : (F, 3, 3) array of floats
        Array with (face, vertex, coords) index coordinates.
    faces : (F, 3) array of ints
        List of length-3 lists of integers, referencing vertex coordinates as
        provided in `verts`.
    spacing : length-3 tuple of floats
        Voxel spacing in spatial dimensions corresponding to numpy array
        indexing dimensions (M, N, P) as in `volume`.
    gradient_direction : string
        Controls if the mesh was generated from an isosurface with gradient
        descent toward objects of interest (the default), or the opposite.
        The two options are:
        * descent : Object was greater than exterior
        * ascent : Exterior was greater than object

    Returns
    -------
    faces_corrected (F, 3) array of ints
        Corrected list of faces referencing vertex coordinates in `verts`.

    Notes
    -----
    Certain applications and mesh processing algorithms require all faces
    to be oriented in a consistent way. Generally, this means a normal vector
    points "out" of the meshed shapes. This algorithm corrects the output from
    `skimage.measure.marching_cubes_classic` by flipping the orientation of
    mis-oriented faces.

    Because marching cubes could be used to find isosurfaces either on
    gradient descent (where the desired object has greater values than the
    exterior) or ascent (where the desired object has lower values than the
    exterior), the ``gradient_direction`` kwarg allows the user to inform this
    algorithm which is correct. If the resulting mesh appears to be oriented
    completely incorrectly, try changing this option.

    The arguments expected by this function are the exact outputs from
    `skimage.measure.marching_cubes_classic` except `actual_verts`, which is an
    uncorrected version of the fancy indexing operation `verts[faces]`.
    Only `faces` is corrected and returned as the vertices do not change,
    only the order in which they are referenced.

    This algorithm assumes ``faces`` provided are exclusively triangles.

    See Also
    --------
    skimage.measure.marching_cubes_classic
    skimage.measure.mesh_surface_area

    """
    # Calculate gradient of `volume`, then interpolate to vertices in `verts`
    grad_x, grad_y, grad_z = np.gradient(volume)

    a = actual_verts[:, 0, :] - actual_verts[:, 1, :]
    b = actual_verts[:, 0, :] - actual_verts[:, 2, :]

    # Find triangle centroids
    centroids = (actual_verts.sum(axis=1) / 3.).T

    del actual_verts

    # Interpolate face centroids into each gradient axis
    grad_centroids_x = ndi.map_coordinates(grad_x, centroids)
    grad_centroids_y = ndi.map_coordinates(grad_y, centroids)
    grad_centroids_z = ndi.map_coordinates(grad_z, centroids)

    # Combine and normalize interpolated gradients
    grad_centroids = np.c_[grad_centroids_x, grad_centroids_y,
                           grad_centroids_z]
    grad_centroids = (grad_centroids /
                      (np.sum(grad_centroids ** 2,
                              axis=1) ** 0.5)[:, np.newaxis])

    # Find normal vectors for each face via cross product
    crosses = np.cross(a, b)
    crosses = crosses / (np.sum(crosses ** 2, axis=1) ** (0.5))[:, np.newaxis]

    # Take dot product
    dotproducts = (grad_centroids * crosses).sum(axis=1)

    # Find mis-oriented faces
    if 'descent' in gradient_direction:
        # Faces with incorrect orientations have dot product < 0
        indices = (dotproducts < 0).nonzero()[0]
    elif 'ascent' in gradient_direction:
        # Faces with incorrection orientation have dot product > 0
        indices = (dotproducts > 0).nonzero()[0]
    else:
        raise ValueError("Incorrect input %s in `gradient_direction`, see "
                         "docstring." % (gradient_direction))

    # Correct orientation and return, without modifying original data
    faces_corrected = faces.copy()
    faces_corrected[indices] = faces_corrected[indices, ::-1]

    return faces_corrected
