import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, convolve

from ..transform import integral_image
from ..feature import structure_tensor
from ..morphology import octagon, star
from ..feature.censure_cy import _censure_dob_loop
from ..feature.util import (FeatureDetector, _prepare_grayscale_input_2D,
                            _mask_border_keypoints)
from .._shared.utils import check_nD

# The paper(Reference [1]) mentions the sizes of the Octagon shaped filter
# kernel for the first seven scales only. The sizes of the later scales
# have been extrapolated based on the following statement in the paper.
# "These octagons scale linearly and were experimentally chosen to correspond
# to the seven DOBs described in the previous section."
OCTAGON_OUTER_SHAPE = [(5, 2), (5, 3), (7, 3), (9, 4), (9, 7), (13, 7),
                       (15, 10), (15, 11), (15, 12), (17, 13), (17, 14)]
OCTAGON_INNER_SHAPE = [(3, 0), (3, 1), (3, 2), (5, 2), (5, 3), (5, 4), (5, 5),
                       (7, 5), (7, 6), (9, 6), (9, 7)]

# The sizes for the STAR shaped filter kernel for different scales have been
# taken from the OpenCV implementation.
STAR_SHAPE = [1, 2, 3, 4, 6, 8, 11, 12, 16, 22, 23, 32, 45, 46, 64, 90, 128]
STAR_FILTER_SHAPE = [(1, 0), (3, 1), (4, 2), (5, 3), (7, 4), (8, 5),
                     (9, 6), (11, 8), (13, 10), (14, 11), (15, 12), (16, 14)]


def _filter_image(image, min_scale, max_scale, mode):

    response = np.zeros((image.shape[0], image.shape[1],
                         max_scale - min_scale + 1), dtype=np.double)

    if mode == 'dob':

        # make response[:, :, i] contiguous memory block
        item_size = response.itemsize
        response.strides = (item_size * response.shape[1], item_size,
                            item_size * response.shape[0] * response.shape[1])

        integral_img = integral_image(image)

        for i in range(max_scale - min_scale + 1):
            n = min_scale + i

            # Constant multipliers for the outer region and the inner region
            # of the bi-level filters with the constraint of keeping the
            # DC bias 0.
            inner_weight = (1.0 / (2 * n + 1) ** 2)
            outer_weight = (1.0 / (12 * n ** 2 + 4 * n))

            _censure_dob_loop(n, integral_img, response[:, :, i],
                              inner_weight, outer_weight)

    # NOTE : For the Octagon shaped filter, we implemented and evaluated the
    # slanted integral image based image filtering but the performance was
    # more or less equal to image filtering using
    # scipy.ndimage.filters.convolve(). Hence we have decided to use the
    # later for a much cleaner implementation.
    elif mode == 'octagon':
        # TODO : Decide the shapes of Octagon filters for scales > 7

        for i in range(max_scale - min_scale + 1):
            mo, no = OCTAGON_OUTER_SHAPE[min_scale + i - 1]
            mi, ni = OCTAGON_INNER_SHAPE[min_scale + i - 1]
            response[:, :, i] = convolve(image,
                                         _octagon_kernel(mo, no, mi, ni))

    elif mode == 'star':

        for i in range(max_scale - min_scale + 1):
            m = STAR_SHAPE[STAR_FILTER_SHAPE[min_scale + i - 1][0]]
            n = STAR_SHAPE[STAR_FILTER_SHAPE[min_scale + i - 1][1]]
            response[:, :, i] = convolve(image, _star_kernel(m, n))

    return response


def _octagon_kernel(mo, no, mi, ni):
    outer = (mo + 2 * no) ** 2 - 2 * no * (no + 1)
    inner = (mi + 2 * ni) ** 2 - 2 * ni * (ni + 1)
    outer_weight = 1.0 / (outer - inner)
    inner_weight = 1.0 / inner
    c = ((mo + 2 * no) - (mi + 2 * ni)) // 2
    outer_oct = octagon(mo, no)
    inner_oct = np.zeros((mo + 2 * no, mo + 2 * no))
    inner_oct[c: -c, c: -c] = octagon(mi, ni)
    bfilter = (outer_weight * outer_oct -
               (outer_weight + inner_weight) * inner_oct)
    return bfilter


def _star_kernel(m, n):
    c = m + m // 2 - n - n // 2
    outer_star = star(m)
    inner_star = np.zeros_like(outer_star)
    inner_star[c: -c, c: -c] = star(n)
    outer_weight = 1.0 / (np.sum(outer_star - inner_star))
    inner_weight = 1.0 / np.sum(inner_star)
    bfilter = (outer_weight * outer_star -
               (outer_weight + inner_weight) * inner_star)
    return bfilter


def _suppress_lines(feature_mask, image, sigma, line_threshold):
    Arr, Arc, Acc = structure_tensor(image, sigma, order='rc')
    feature_mask[(Arr + Acc) ** 2
                 > line_threshold * (Arr * Acc - Arc ** 2)] = False


class CENSURE(FeatureDetector):

    """CENSURE keypoint detector.

    min_scale : int, optional
        Minimum scale to extract keypoints from.
    max_scale : int, optional
        Maximum scale to extract keypoints from. The keypoints will be
        extracted from all the scales except the first and the last i.e.
        from the scales in the range [min_scale + 1, max_scale - 1]. The filter
        sizes for different scales is such that the two adjacent scales
        comprise of an octave.
    mode : {'DoB', 'Octagon', 'STAR'}, optional
        Type of bi-level filter used to get the scales of the input image.
        Possible values are 'DoB', 'Octagon' and 'STAR'. The three modes
        represent the shape of the bi-level filters i.e. box(square), octagon
        and star respectively. For instance, a bi-level octagon filter consists
        of a smaller inner octagon and a larger outer octagon with the filter
        weights being uniformly negative in both the inner octagon while
        uniformly positive in the difference region. Use STAR and Octagon for
        better features and DoB for better performance.
    non_max_threshold : float, optional
        Threshold value used to suppress maximas and minimas with a weak
        magnitude response obtained after Non-Maximal Suppression.
    line_threshold : float, optional
        Threshold for rejecting interest points which have ratio of principal
        curvatures greater than this value.

    Attributes
    ----------
    keypoints : (N, 2) array
        Keypoint coordinates as ``(row, col)``.
    scales : (N, ) array
        Corresponding scales.

    References
    ----------
    .. [1] Motilal Agrawal, Kurt Konolige and Morten Rufus Blas
           "CENSURE: Center Surround Extremas for Realtime Feature
           Detection and Matching",
           https://link.springer.com/chapter/10.1007/978-3-540-88693-8_8
           :DOI:`10.1007/978-3-540-88693-8_8`

    .. [2] Adam Schmidt, Marek Kraft, Michal Fularz and Zuzanna Domagala
           "Comparative Assessment of Point Feature Detectors and
           Descriptors in the Context of Robot Navigation"
           http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-268aaf28-0faf-4872-a4df-7e2e61cb364c/c/Schmidt_comparative.pdf
           :DOI:`10.1.1.465.1117`

    Examples
    --------
    >>> from skimage.data import astronaut
    >>> from skimage.color import rgb2gray
    >>> from skimage.feature import CENSURE
    >>> img = rgb2gray(astronaut()[100:300, 100:300])
    >>> censure = CENSURE()
    >>> censure.detect(img)
    >>> censure.keypoints
    array([[  4, 148],
           [ 12,  73],
           [ 21, 176],
           [ 91,  22],
           [ 93,  56],
           [ 94,  22],
           [ 95,  54],
           [100,  51],
           [103,  51],
           [106,  67],
           [108,  15],
           [117,  20],
           [122,  60],
           [125,  37],
           [129,  37],
           [133,  76],
           [145,  44],
           [146,  94],
           [150, 114],
           [153,  33],
           [154, 156],
           [155, 151],
           [184,  63]])
    >>> censure.scales
    array([2, 6, 6, 2, 4, 3, 2, 3, 2, 6, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 4, 2,
           2])

    """

    def __init__(self, min_scale=1, max_scale=7, mode='DoB',
                 non_max_threshold=0.15, line_threshold=10):

        mode = mode.lower()
        if mode not in ('dob', 'octagon', 'star'):
            raise ValueError("`mode` must be one of 'DoB', 'Octagon', 'STAR'.")

        if min_scale < 1 or max_scale < 1 or max_scale - min_scale < 2:
            raise ValueError('The scales must be >= 1 and the number of '
                             'scales should be >= 3.')

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mode = mode
        self.non_max_threshold = non_max_threshold
        self.line_threshold = line_threshold

        self.keypoints = None
        self.scales = None

    def detect(self, image):
        """Detect CENSURE keypoints along with the corresponding scale.

        Parameters
        ----------
        image : 2D ndarray
            Input image.

        """

        # (1) First we generate the required scales on the input grayscale
        # image using a bi-level filter and stack them up in `filter_response`.

        # (2) We then perform Non-Maximal suppression in 3 x 3 x 3 window on
        # the filter_response to suppress points that are neither minima or
        # maxima in 3 x 3 x 3 neighbourhood. We obtain a boolean ndarray
        # `feature_mask` containing all the minimas and maximas in
        # `filter_response` as True.
        # (3) Then we suppress all the points in the `feature_mask` for which
        # the corresponding point in the image at a particular scale has the
        # ratio of principal curvatures greater than `line_threshold`.
        # (4) Finally, we remove the border keypoints and return the keypoints
        # along with its corresponding scale.

        check_nD(image, 2)

        num_scales = self.max_scale - self.min_scale

        image = np.ascontiguousarray(_prepare_grayscale_input_2D(image))

        # Generating all the scales
        filter_response = _filter_image(image, self.min_scale, self.max_scale,
                                        self.mode)

        # Suppressing points that are neither minima or maxima in their
        # 3 x 3 x 3 neighborhood to zero
        minimas = minimum_filter(filter_response, (3, 3, 3)) == filter_response
        maximas = maximum_filter(filter_response, (3, 3, 3)) == filter_response

        feature_mask = minimas | maximas
        feature_mask[filter_response < self.non_max_threshold] = False

        for i in range(1, num_scales):
            # sigma = (window_size - 1) / 6.0, so the window covers > 99% of
            #                                  the kernel's distribution
            # window_size = 7 + 2 * (min_scale - 1 + i)
            # Hence sigma = 1 + (min_scale - 1 + i)/ 3.0
            _suppress_lines(feature_mask[:, :, i], image,
                            (1 + (self.min_scale + i - 1) / 3.0),
                            self.line_threshold)

        rows, cols, scales = np.nonzero(feature_mask[..., 1:num_scales])
        keypoints = np.column_stack([rows, cols])
        scales = scales + self.min_scale + 1

        if self.mode == 'dob':
            self.keypoints = keypoints
            self.scales = scales
            return

        cumulative_mask = np.zeros(keypoints.shape[0], dtype=bool)

        if self.mode == 'octagon':
            for i in range(self.min_scale + 1, self.max_scale):
                c = (OCTAGON_OUTER_SHAPE[i - 1][0] - 1) // 2 \
                    + OCTAGON_OUTER_SHAPE[i - 1][1]
                cumulative_mask |= (
                    _mask_border_keypoints(image.shape, keypoints, c)
                    & (scales == i))
        elif self.mode == 'star':
            for i in range(self.min_scale + 1, self.max_scale):
                c = STAR_SHAPE[STAR_FILTER_SHAPE[i - 1][0]] \
                    + STAR_SHAPE[STAR_FILTER_SHAPE[i - 1][0]] // 2
                cumulative_mask |= (
                    _mask_border_keypoints(image.shape, keypoints, c)
                    & (scales == i))

        self.keypoints = keypoints[cumulative_mask]
        self.scales = scales[cumulative_mask]
