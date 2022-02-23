"""
Methods to characterize image textures.
"""

import numpy as np

from .._shared.utils import check_nD
from ..color import gray2rgb
from ..util import img_as_float
from ._texture import (_glcm_loop,
                       _local_binary_pattern,
                       _multiblock_lbp)


def graycomatrix(image, distances, angles, levels=None, symmetric=False,
                 normed=False):
    """Calculate the gray-level co-occurrence matrix.

    A gray level co-occurrence matrix is a histogram of co-occurring
    grayscale values at a given offset over an image.

    Parameters
    ----------
    image : array_like
        Integer typed input image. Only positive valued images are supported.
        If type is other than uint8, the argument `levels` needs to be set.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, `levels`-1],
        where levels indicate the number of gray-levels counted
        (typically 256 for an 8-bit image). This argument is required for
        16-bit images or higher and is typically the maximum of the image.
        As the output matrix is at least `levels` x `levels`, it might
        be preferable to use binning of the input image rather than
        large values for `levels`.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.

    Returns
    -------
    P : 4-D ndarray
        The gray-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that gray-level `j`
        occurs at a distance `d` and at an angle `theta` from
        gray-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64. The dimensions are:
        levels x levels x number of distances x number of angles.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`
    .. [2] R.M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for
           image classification", IEEE Transactions on Systems, Man, and
           Cybernetics, vol. SMC-3, no. 6, pp. 610-621, Nov. 1973.
           :DOI:`10.1109/TSMC.1973.4309314`
    .. [3] M. Nadler and E.P. Smith, Pattern Recognition Engineering,
           Wiley-Interscience, 1993.
    .. [4] Wikipedia, https://en.wikipedia.org/wiki/Co-occurrence_matrix


    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
    ...                       levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)

    """
    check_nD(image, 2)
    check_nD(distances, 1, 'distances')
    check_nD(angles, 1, 'angles')

    image = np.ascontiguousarray(image)

    image_max = image.max()

    if np.issubdtype(image.dtype, np.floating):
        raise ValueError("Float images are not supported by graycomatrix. "
                         "Convert the image to an unsigned integer type.")

    # for image type > 8bit, levels must be set.
    if image.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError("The levels argument is required for data types "
                         "other than uint8. The resulting matrix will be at "
                         "least levels ** 2 in size.")

    if np.issubdtype(image.dtype, np.signedinteger) and np.any(image < 0):
        raise ValueError("Negative-valued images are not supported.")

    if levels is None:
        levels = 256

    if image_max >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")

    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)

    P = np.zeros((levels, levels, len(distances), len(angles)),
                 dtype=np.uint32, order='C')

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P)

    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt

    # normalize each GLCM
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P


def graycoprops(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    Compute a feature of a gray level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]

    Each GLCM is normalized to have a sum of 1 before the computation of
    texture properties.

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM'}, optional
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = graycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])

    """
    check_nD(P, 4, 'P')

    (num_level, num_level2, num_dist, num_angle) = P.shape
    if num_level != num_level2:
        raise ValueError('num_level and num_level2 must be equal.')
    if num_dist <= 0:
        raise ValueError('num_dist must be positive.')
    if num_angle <= 0:
        raise ValueError('num_angle must be positive.')

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums = np.sum(P, axis=(0, 1), keepdims=True)
    glcm_sums[glcm_sums == 0] = 1
    P /= glcm_sums

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    else:
        raise ValueError('%s is an invalid property' % (prop))

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.sum(P ** 2, axis=(0, 1))
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.sum(P ** 2, axis=(0, 1))
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.sum(I * P, axis=(0, 1))
        diff_j = J - np.sum(J * P, axis=(0, 1))

        std_i = np.sqrt(np.sum(P * (diff_i) ** 2, axis=(0, 1)))
        std_j = np.sqrt(np.sum(P * (diff_j) ** 2, axis=(0, 1)))
        cov = np.sum(P * (diff_i * diff_j), axis=(0, 1))

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = ~mask_0
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop in ['contrast', 'dissimilarity', 'homogeneity']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.sum(P * weights, axis=(0, 1))

    return results


def local_binary_pattern(image, P, R, method='default'):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.

        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
        * 'uniform': improved rotation invariance with uniform patterns and
            finer quantization of the angular space which is gray scale and
            rotation invariant.
        * 'nri_uniform': non rotation-invariant uniform patterns variant
            which is only gray scale invariant [2]_, [3]_.
        * 'var': rotation invariant variance measures of the contrast of local
            image texture which is rotation but not gray scale invariant.

    Returns
    -------
    output : (N, M) array
        LBP image.

    References
    ----------
    .. [1] T. Ojala, M. Pietikainen, T. Maenpaa, "Multiresolution gray-scale
           and rotation invariant texture classification with local binary
           patterns", IEEE Transactions on Pattern Analysis and Machine
           Intelligence, vol. 24, no. 7, pp. 971-987, July 2002
           :DOI:`10.1109/TPAMI.2002.1017623`
    .. [2] T. Ahonen, A. Hadid and M. Pietikainen. "Face recognition with
           local binary patterns", in Proc. Eighth European Conf. Computer
           Vision, Prague, Czech Republic, May 11-14, 2004, pp. 469-481, 2004.
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851
           :DOI:`10.1007/978-3-540-24670-1_36`
    .. [3] T. Ahonen, A. Hadid and M. Pietikainen, "Face Description with
           Local Binary Patterns: Application to Face Recognition",
           IEEE Transactions on Pattern Analysis and Machine Intelligence,
           vol. 28, no. 12, pp. 2037-2041, Dec. 2006
           :DOI:`10.1109/TPAMI.2006.244.`
    """
    check_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output


def multiblock_lbp(int_image, r, c, width, height):
    """Multi-block local binary pattern (MB-LBP).

    The features are calculated similarly to local binary patterns (LBPs),
    (See :py:meth:`local_binary_pattern`) except that summed blocks are
    used instead of individual pixel values.

    MB-LBP is an extension of LBP that can be computed on multiple scales
    in constant time using the integral image. Nine equally-sized rectangles
    are used to compute a feature. For each rectangle, the sum of the pixel
    intensities is computed. Comparisons of these sums to that of the central
    rectangle determine the feature, similarly to LBP.

    Parameters
    ----------
    int_image : (N, M) array
        Integral image.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of the 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of the 9 equal rectangles that will be used to compute
        a feature.

    Returns
    -------
    output : int
        8-bit MB-LBP feature descriptor.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    """

    int_image = np.ascontiguousarray(int_image, dtype=np.float32)
    lbp_code = _multiblock_lbp(int_image, r, c, width, height)
    return lbp_code


def draw_multiblock_lbp(image, r, c, width, height,
                        lbp_code=0,
                        color_greater_block=(1, 1, 1),
                        color_less_block=(0, 0.69, 0.96),
                        alpha=0.5
                        ):
    """Multi-block local binary pattern visualization.

    Blocks with higher sums are colored with alpha-blended white rectangles,
    whereas blocks with lower sums are colored alpha-blended cyan. Colors
    and the `alpha` parameter can be changed.

    Parameters
    ----------
    image : ndarray of float or uint
        Image on which to visualize the pattern.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of 9 equal rectangles that will be used to compute
        a feature.
    lbp_code : int
        The descriptor of feature to visualize. If not provided, the
        descriptor with 0 value will be used.
    color_greater_block : tuple of 3 floats
        Floats specifying the color for the block that has greater
        intensity value. They should be in the range [0, 1].
        Corresponding values define (R, G, B) values. Default value
        is white (1, 1, 1).
    color_greater_block : tuple of 3 floats
        Floats specifying the color for the block that has greater intensity
        value. They should be in the range [0, 1]. Corresponding values define
        (R, G, B) values. Default value is cyan (0, 0.69, 0.96).
    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization.
        1 - fully transparent, 0 - opaque.

    Returns
    -------
    output : ndarray of float
        Image with MB-LBP visualization.

    References
    ----------
    .. [1] L. Zhang, R. Chu, S. Xiang, S. Liao, S.Z. Li. "Face Detection Based
           on Multi-Block LBP Representation", In Proceedings: Advances in
           Biometrics, International Conference, ICB 2007, Seoul, Korea.
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
           :DOI:`10.1007/978-3-540-74549-5_2`
    """

    # Default colors for regions.
    # White is for the blocks that are brighter.
    # Cyan is for the blocks that has less intensity.
    color_greater_block = np.asarray(color_greater_block, dtype=np.float64)
    color_less_block = np.asarray(color_less_block, dtype=np.float64)

    # Copy array to avoid the changes to the original one.
    output = np.copy(image)

    # As the visualization uses RGB color we need 3 bands.
    if len(image.shape) < 3:
        output = gray2rgb(image)

    # Colors are specified in floats.
    output = img_as_float(output)

    # Offsets of neighbour rectangles relative to central one.
    # It has order starting from top left and going clockwise.
    neighbour_rect_offsets = ((-1, -1), (-1, 0), (-1, 1),
                              (0, 1), (1, 1), (1, 0),
                              (1, -1), (0, -1))

    # Pre-multiply the offsets with width and height.
    neighbour_rect_offsets = np.array(neighbour_rect_offsets)
    neighbour_rect_offsets[:, 0] *= height
    neighbour_rect_offsets[:, 1] *= width

    # Top-left coordinates of central rectangle.
    central_rect_r = r + height
    central_rect_c = c + width

    for element_num, offset in enumerate(neighbour_rect_offsets):

        offset_r, offset_c = offset

        curr_r = central_rect_r + offset_r
        curr_c = central_rect_c + offset_c

        has_greater_value = lbp_code & (1 << (7-element_num))

        # Mix-in the visualization colors.
        if has_greater_value:
            new_value = ((1-alpha) *
                         output[curr_r:curr_r+height, curr_c:curr_c+width] +
                         alpha * color_greater_block)
            output[curr_r:curr_r+height, curr_c:curr_c+width] = new_value
        else:
            new_value = ((1-alpha) *
                         output[curr_r:curr_r+height, curr_c:curr_c+width] +
                         alpha * color_less_block)
            output[curr_r:curr_r+height, curr_c:curr_c+width] = new_value

    return output
