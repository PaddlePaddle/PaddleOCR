import numpy as np

from ..util import img_as_float
from .._shared.utils import _supported_float_type, check_nD


class FeatureDetector(object):

    def __init__(self):
        self.keypoints_ = np.array([])

    def detect(self, image):
        """Detect keypoints in image.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        raise NotImplementedError()


class DescriptorExtractor(object):

    def __init__(self):
        self.descriptors_ = np.array([])

    def extract(self, image, keypoints):
        """Extract feature descriptors in image for given keypoints.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint locations as ``(row, col)``.

        """
        raise NotImplementedError()


def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color='k', matches_color=None, only_matches=False,
                 alignment='horizontal'):
    """Plot matched features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.

    """
    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    offset = np.array(image1.shape)
    if alignment == 'horizontal':
        image = np.concatenate([image1, image2], axis=1)
        offset[0] = 0
    elif alignment == 'vertical':
        image = np.concatenate([image1, image2], axis=0)
        offset[1] = 0
    else:
        mesg = (f"plot_matches accepts either 'horizontal' or 'vertical' for "
                f"alignment, but '{alignment}' was given. See "
                f"https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.plot_matches "  # noqa
                f"for details.")
        raise ValueError(mesg)

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0] + offset[0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image, cmap='gray')
    ax.axis((0, image1.shape[1] + offset[1], image1.shape[0] + offset[0], 0))

    rng = np.random.default_rng()

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        if matches_color is None:
            color = rng.random(3)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                (keypoints1[idx1, 0], keypoints2[idx2, 0] + offset[0]),
                '-', color=color)


def _prepare_grayscale_input_2D(image):
    image = np.squeeze(image)
    check_nD(image, 2)
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)


def _prepare_grayscale_input_nD(image):
    image = np.squeeze(image)
    check_nD(image, range(2, 6))
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    return image.astype(float_dtype, copy=False)


def _mask_border_keypoints(image_shape, keypoints, distance):
    """Mask coordinates that are within certain distance from the image border.

    Parameters
    ----------
    image_shape : (2, ) array_like
        Shape of the image as ``(rows, cols)``.
    keypoints : (N, 2) array
        Keypoint coordinates as ``(rows, cols)``.
    distance : int
        Image border distance.

    Returns
    -------
    mask : (N, ) bool array
        Mask indicating if pixels are within the image (``True``) or in the
        border region of the image (``False``).

    """

    rows = image_shape[0]
    cols = image_shape[1]

    mask = (((distance - 1) < keypoints[:, 0])
            & (keypoints[:, 0] < (rows - distance + 1))
            & ((distance - 1) < keypoints[:, 1])
            & (keypoints[:, 1] < (cols - distance + 1)))

    return mask
