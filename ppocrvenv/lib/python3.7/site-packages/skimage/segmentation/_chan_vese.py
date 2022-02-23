import numpy as np
from scipy.ndimage import distance_transform_edt as distance

from .._shared.utils import _supported_float_type, deprecate_kwarg


def _cv_curvature(phi):
    """Returns the 'curvature' of a level set 'phi'.
    """
    P = np.pad(phi, 1, mode='edge')
    fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    fyy = P[2:, 1:-1] + P[:-2, 1:-1] - 2*phi
    fxx = P[1:-1, 2:] + P[1:-1, :-2] - 2*phi
    fxy = .25 * (P[2:, 2:] + P[:-2, :-2] - P[:-2, 2:] - P[2:, :-2])
    grad2 = fx**2 + fy**2
    K = ((fxx*fy**2 - 2*fxy*fx*fy + fyy*fx**2) /
         (grad2*np.sqrt(grad2) + 1e-8))
    return K


def _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt):
    """Returns the variation of level set 'phi' based on algorithm parameters.
    """
    eta = 1e-16
    P = np.pad(phi, 1, mode='edge')

    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
    phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0

    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
    phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0

    C1 = 1. / np.sqrt(eta + phixp**2 + phiy0**2)
    C2 = 1. / np.sqrt(eta + phixn**2 + phiy0**2)
    C3 = 1. / np.sqrt(eta + phix0**2 + phiyp**2)
    C4 = 1. / np.sqrt(eta + phix0**2 + phiyn**2)

    K = (P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 +
         P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4)

    Hphi = 1 * (phi > 0)
    (c1, c2) = _cv_calculate_averages(image, Hphi)

    difference_from_average_term = (- lambda1 * (image-c1)**2 +
                                    lambda2 * (image-c2)**2)
    new_phi = (phi + (dt*_cv_delta(phi)) *
               (mu*K + difference_from_average_term))
    return new_phi / (1 + mu * dt * _cv_delta(phi) * (C1+C2+C3+C4))


def _cv_heavyside(x, eps=1.):
    """Returns the result of a regularised heavyside function of the
    input value(s).
    """
    return 0.5 * (1. + (2./np.pi) * np.arctan(x/eps))


def _cv_delta(x, eps=1.):
    """Returns the result of a regularised dirac function of the
    input value(s).
    """
    return eps / (eps**2 + x**2)


def _cv_calculate_averages(image, Hphi):
    """Returns the average values 'inside' and 'outside'.
    """
    H = Hphi
    Hinv = 1. - H
    Hsum = np.sum(H)
    Hinvsum = np.sum(Hinv)
    avg_inside = np.sum(image * H)
    avg_oustide = np.sum(image * Hinv)
    if Hsum != 0:
        avg_inside /= Hsum
    if Hinvsum != 0:
        avg_oustide /= Hinvsum
    return (avg_inside, avg_oustide)


def _cv_difference_from_average_term(image, Hphi, lambda_pos, lambda_neg):
    """Returns the 'energy' contribution due to the difference from
    the average value within a region at each point.
    """
    (c1, c2) = _cv_calculate_averages(image, Hphi)
    Hinv = 1. - Hphi
    return (lambda_pos * (image-c1)**2 * Hphi +
            lambda_neg * (image-c2)**2 * Hinv)


def _cv_edge_length_term(phi, mu):
    """Returns the 'energy' contribution due to the length of the
    edge between regions at each point, multiplied by a factor 'mu'.
    """
    toret = _cv_curvature(phi)
    return mu * toret


def _cv_energy(image, phi, mu, lambda1, lambda2):
    """Returns the total 'energy' of the current level set function.
    """
    H = _cv_heavyside(phi)
    avgenergy = _cv_difference_from_average_term(image, H, lambda1, lambda2)
    lenenergy = _cv_edge_length_term(phi, mu)
    return np.sum(avgenergy) + np.sum(lenenergy)


def _cv_reset_level_set(phi):
    """This is a placeholder function as resetting the level set is not
    strictly necessary, and has not been done for this implementation.
    """
    return phi


def _cv_checkerboard(image_size, square_size, dtype=np.float64):
    """Generates a checkerboard level set function.

    According to Pascal Getreuer, such a level set function has fast
    convergence.
    """
    yv = np.arange(image_size[0], dtype=dtype).reshape(image_size[0], 1)
    xv = np.arange(image_size[1], dtype=dtype)
    sf = np.pi / square_size
    xv *= sf
    yv *= sf
    return np.sin(yv) * np.sin(xv)


def _cv_large_disk(image_size):
    """Generates a disk level set function.

    The disk covers the whole image along its smallest dimension.
    """
    res = np.ones(image_size)
    centerY = int((image_size[0]-1) / 2)
    centerX = int((image_size[1]-1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY))
    return (radius - distance(res)) / radius


def _cv_small_disk(image_size):
    """Generates a disk level set function.

    The disk covers half of the image along its smallest dimension.
    """
    res = np.ones(image_size)
    centerY = int((image_size[0]-1) / 2)
    centerX = int((image_size[1]-1) / 2)
    res[centerY, centerX] = 0.
    radius = float(min(centerX, centerY)) / 2.0
    return (radius - distance(res)) / (radius * 3)


def _cv_init_level_set(init_level_set, image_shape, dtype=np.float64):
    """Generates an initial level set function conditional on input arguments.
    """
    if type(init_level_set) == str:
        if init_level_set == 'checkerboard':
            res = _cv_checkerboard(image_shape, 5, dtype)
        elif init_level_set == 'disk':
            res = _cv_large_disk(image_shape)
        elif init_level_set == 'small disk':
            res = _cv_small_disk(image_shape)
        else:
            raise ValueError("Incorrect name for starting level set preset.")
    else:
        res = init_level_set
    return res.astype(dtype, copy=False)


@deprecate_kwarg({'max_iter': 'max_num_iter'}, removed_version="1.0",
                 deprecated_version="0.19")
def chan_vese(image, mu=0.25, lambda1=1.0, lambda2=1.0, tol=1e-3,
              max_num_iter=500, dt=0.5, init_level_set='checkerboard',
              extended_output=False):
    """Chan-Vese segmentation algorithm.

    Active contour model by evolving a level set. Can be used to
    segment objects without clearly defined boundaries.

    Parameters
    ----------
    image : (M, N) ndarray
        Grayscale image to be segmented.
    mu : float, optional
        'edge length' weight parameter. Higher `mu` values will
        produce a 'round' edge, while values closer to zero will
        detect smaller objects.
    lambda1 : float, optional
        'difference from average' weight parameter for the output
        region with value 'True'. If it is lower than `lambda2`, this
        region will have a larger range of values than the other.
    lambda2 : float, optional
        'difference from average' weight parameter for the output
        region with value 'False'. If it is lower than `lambda1`, this
        region will have a larger range of values than the other.
    tol : float, positive, optional
        Level set variation tolerance between iterations. If the
        L2 norm difference between the level sets of successive
        iterations normalized by the area of the image is below this
        value, the algorithm will assume that the solution was
        reached.
    max_num_iter : uint, optional
        Maximum number of iterations allowed before the algorithm
        interrupts itself.
    dt : float, optional
        A multiplication factor applied at calculations for each step,
        serves to accelerate the algorithm. While higher values may
        speed up the algorithm, they may also lead to convergence
        problems.
    init_level_set : str or (M, N) ndarray, optional
        Defines the starting level set used by the algorithm.
        If a string is inputted, a level set that matches the image
        size will automatically be generated. Alternatively, it is
        possible to define a custom level set, which should be an
        array of float values, with the same shape as 'image'.
        Accepted string values are as follows.

        'checkerboard'
            the starting level set is defined as
            sin(x/5*pi)*sin(y/5*pi), where x and y are pixel
            coordinates. This level set has fast convergence, but may
            fail to detect implicit edges.
        'disk'
            the starting level set is defined as the opposite
            of the distance from the center of the image minus half of
            the minimum value between image width and image height.
            This is somewhat slower, but is more likely to properly
            detect implicit edges.
        'small disk'
            the starting level set is defined as the
            opposite of the distance from the center of the image
            minus a quarter of the minimum value between image width
            and image height.
    extended_output : bool, optional
        If set to True, the return value will be a tuple containing
        the three return values (see below). If set to False which
        is the default value, only the 'segmentation' array will be
        returned.

    Returns
    -------
    segmentation : (M, N) ndarray, bool
        Segmentation produced by the algorithm.
    phi : (M, N) ndarray of floats
        Final level set computed by the algorithm.
    energies : list of floats
        Shows the evolution of the 'energy' for each step of the
        algorithm. This should allow to check whether the algorithm
        converged.

    Notes
    -----
    The Chan-Vese Algorithm is designed to segment objects without
    clearly defined boundaries. This algorithm is based on level sets
    that are evolved iteratively to minimize an energy, which is
    defined by weighted values corresponding to the sum of differences
    intensity from the average value outside the segmented region, the
    sum of differences from the average value inside the segmented
    region, and a term which is dependent on the length of the
    boundary of the segmented region.

    This algorithm was first proposed by Tony Chan and Luminita Vese,
    in a publication entitled "An Active Contour Model Without Edges"
    [1]_.

    This implementation of the algorithm is somewhat simplified in the
    sense that the area factor 'nu' described in the original paper is
    not implemented, and is only suitable for grayscale images.

    Typical values for `lambda1` and `lambda2` are 1. If the
    'background' is very different from the segmented object in terms
    of distribution (for example, a uniform black image with figures
    of varying intensity), then these values should be different from
    each other.

    Typical values for mu are between 0 and 1, though higher values
    can be used when dealing with shapes with very ill-defined
    contours.

    The 'energy' which this algorithm tries to minimize is defined
    as the sum of the differences from the average within the region
    squared and weighed by the 'lambda' factors to which is added the
    length of the contour multiplied by the 'mu' factor.

    Supports 2D grayscale images only, and does not implement the area
    term described in the original article.

    References
    ----------
    .. [1] An Active Contour Model without Edges, Tony Chan and
           Luminita Vese, Scale-Space Theories in Computer Vision,
           1999, :DOI:`10.1007/3-540-48236-9_13`
    .. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On
           Line, 2 (2012), pp. 214-224,
           :DOI:`10.5201/ipol.2012.g-cv`
    .. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen, 2011
           :arXiv:`1107.2782`
    """
    if len(image.shape) != 2:
        raise ValueError("Input image should be a 2D array.")

    float_dtype = _supported_float_type(image.dtype)
    phi = _cv_init_level_set(init_level_set, image.shape, dtype=float_dtype)

    if type(phi) != np.ndarray or phi.shape != image.shape:
        raise ValueError("The dimensions of initial level set do not "
                         "match the dimensions of image.")

    image = image.astype(float_dtype, copy=False)
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)

    i = 0
    old_energy = _cv_energy(image, phi, mu, lambda1, lambda2)
    energies = []
    phivar = tol + 1
    segmentation = phi > 0

    while(phivar > tol and i < max_num_iter):
        # Save old level set values
        oldphi = phi

        # Calculate new level set
        phi = _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt)
        phi = _cv_reset_level_set(phi)
        phivar = np.sqrt(((phi-oldphi)**2).mean())

        # Extract energy and compare to previous level set and
        # segmentation to see if continuing is necessary
        segmentation = phi > 0
        new_energy = _cv_energy(image, phi, mu, lambda1, lambda2)

        # Save old energy values
        energies.append(old_energy)
        old_energy = new_energy
        i += 1

    if extended_output:
        return (segmentation, phi, energies)
    else:
        return segmentation
