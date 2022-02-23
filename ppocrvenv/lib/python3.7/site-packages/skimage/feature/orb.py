import numpy as np

from ..feature.util import (FeatureDetector, DescriptorExtractor,
                            _mask_border_keypoints,
                            _prepare_grayscale_input_2D)

from ..feature import (corner_fast, corner_orientations, corner_peaks,
                       corner_harris)
from ..transform import pyramid_gaussian
from .._shared.utils import check_nD

from .orb_cy import _orb_loop


OFAST_MASK = np.zeros((31, 31))
OFAST_UMAX = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
for i in range(-15, 16):
    for j in range(-OFAST_UMAX[abs(i)], OFAST_UMAX[abs(i)] + 1):
        OFAST_MASK[15 + j, 15 + i] = 1


class ORB(FeatureDetector, DescriptorExtractor):

    """Oriented FAST and rotated BRIEF feature detector and binary descriptor
    extractor.

    Parameters
    ----------
    n_keypoints : int, optional
        Number of keypoints to be returned. The function will return the best
        `n_keypoints` according to the Harris corner response if more than
        `n_keypoints` are detected. If not, then all the detected keypoints
        are returned.
    fast_n : int, optional
        The `n` parameter in `skimage.feature.corner_fast`. Minimum number of
        consecutive pixels out of 16 pixels on the circle that should all be
        either brighter or darker w.r.t test-pixel. A point c on the circle is
        darker w.r.t test pixel p if ``Ic < Ip - threshold`` and brighter if
        ``Ic > Ip + threshold``. Also stands for the n in ``FAST-n`` corner
        detector.
    fast_threshold : float, optional
        The ``threshold`` parameter in ``feature.corner_fast``. Threshold used
        to decide whether the pixels on the circle are brighter, darker or
        similar w.r.t. the test pixel. Decrease the threshold when more
        corners are desired and vice-versa.
    harris_k : float, optional
        The `k` parameter in `skimage.feature.corner_harris`. Sensitivity
        factor to separate corners from edges, typically in range ``[0, 0.2]``.
        Small values of `k` result in detection of sharp corners.
    downscale : float, optional
        Downscale factor for the image pyramid. Default value 1.2 is chosen so
        that there are more dense scales which enable robust scale invariance
        for a subsequent feature description.
    n_scales : int, optional
        Maximum number of scales from the bottom of the image pyramid to
        extract the features from.

    Attributes
    ----------
    keypoints : (N, 2) array
        Keypoint coordinates as ``(row, col)``.
    scales : (N, ) array
        Corresponding scales.
    orientations : (N, ) array
        Corresponding orientations in radians.
    responses : (N, ) array
        Corresponding Harris corner responses.
    descriptors : (Q, `descriptor_size`) array of dtype bool
        2D array of binary descriptors of size `descriptor_size` for Q
        keypoints after filtering out border keypoints with value at an
        index ``(i, j)`` either being ``True`` or ``False`` representing
        the outcome of the intensity comparison for i-th keypoint on j-th
        decision pixel-pair. It is ``Q == np.sum(mask)``.

    References
    ----------
    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB: An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf

    Examples
    --------
    >>> from skimage.feature import ORB, match_descriptors
    >>> img1 = np.zeros((100, 100))
    >>> img2 = np.zeros_like(img1)
    >>> rng = np.random.default_rng(19481137)  # do not copy this value
    >>> square = rng.random((20, 20))
    >>> img1[40:60, 40:60] = square
    >>> img2[53:73, 53:73] = square
    >>> detector_extractor1 = ORB(n_keypoints=5)
    >>> detector_extractor2 = ORB(n_keypoints=5)
    >>> detector_extractor1.detect_and_extract(img1)
    >>> detector_extractor2.detect_and_extract(img2)
    >>> matches = match_descriptors(detector_extractor1.descriptors,
    ...                             detector_extractor2.descriptors)
    >>> matches
    array([[0, 0],
           [1, 1],
           [2, 2],
           [3, 4],
           [4, 3]])
    >>> detector_extractor1.keypoints[matches[:, 0]]
    array([[59. , 59. ],
           [40. , 40. ],
           [57. , 40. ],
           [46. , 58. ],
           [58.8, 58.8]])
    >>> detector_extractor2.keypoints[matches[:, 1]]
    array([[72., 72.],
           [53., 53.],
           [70., 53.],
           [59., 71.],
           [72., 72.]])

    """

    def __init__(self, downscale=1.2, n_scales=8,
                 n_keypoints=500, fast_n=9, fast_threshold=0.08,
                 harris_k=0.04):
        self.downscale = downscale
        self.n_scales = n_scales
        self.n_keypoints = n_keypoints
        self.fast_n = fast_n
        self.fast_threshold = fast_threshold
        self.harris_k = harris_k

        self.keypoints = None
        self.scales = None
        self.responses = None
        self.orientations = None
        self.descriptors = None

    def _build_pyramid(self, image):
        image = _prepare_grayscale_input_2D(image)
        return list(pyramid_gaussian(image, self.n_scales - 1,
                                     self.downscale, channel_axis=None))

    def _detect_octave(self, octave_image):
        dtype = octave_image.dtype
        # Extract keypoints for current octave
        fast_response = corner_fast(octave_image, self.fast_n,
                                    self.fast_threshold)
        keypoints = corner_peaks(fast_response, min_distance=1)

        if len(keypoints) == 0:
            return (np.zeros((0, 2), dtype=dtype),
                    np.zeros((0, ), dtype=dtype),
                    np.zeros((0, ), dtype=dtype))

        mask = _mask_border_keypoints(octave_image.shape, keypoints,
                                      distance=16)
        keypoints = keypoints[mask]

        orientations = corner_orientations(octave_image, keypoints,
                                           OFAST_MASK)

        harris_response = corner_harris(octave_image, method='k',
                                        k=self.harris_k)
        responses = harris_response[keypoints[:, 0], keypoints[:, 1]]

        return keypoints, orientations, responses

    def detect(self, image):
        """Detect oriented FAST keypoints along with the corresponding scale.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)

        pyramid = self._build_pyramid(image)

        keypoints_list = []
        orientations_list = []
        scales_list = []
        responses_list = []

        for octave in range(len(pyramid)):

            octave_image = np.ascontiguousarray(pyramid[octave])

            keypoints, orientations, responses = self._detect_octave(
                octave_image)

            keypoints_list.append(keypoints * self.downscale ** octave)
            orientations_list.append(orientations)
            scales_list.append(np.full(
                keypoints.shape[0], self.downscale ** octave,
                dtype=octave_image.dtype))
            responses_list.append(responses)

        keypoints = np.vstack(keypoints_list)
        orientations = np.hstack(orientations_list)
        scales = np.hstack(scales_list)
        responses = np.hstack(responses_list)

        if keypoints.shape[0] < self.n_keypoints:
            self.keypoints = keypoints
            self.scales = scales
            self.orientations = orientations
            self.responses = responses
        else:
            # Choose best n_keypoints according to Harris corner response
            best_indices = responses.argsort()[::-1][:self.n_keypoints]
            self.keypoints = keypoints[best_indices]
            self.scales = scales[best_indices]
            self.orientations = orientations[best_indices]
            self.responses = responses[best_indices]

    def _extract_octave(self, octave_image, keypoints, orientations):
        mask = _mask_border_keypoints(octave_image.shape, keypoints,
                                      distance=20)
        keypoints = np.array(keypoints[mask], dtype=np.intp, order='C',
                             copy=False)
        orientations = np.array(orientations[mask], order='C',
                                copy=False)

        descriptors = _orb_loop(octave_image, keypoints, orientations)

        return descriptors, mask

    def extract(self, image, keypoints, scales, orientations):
        """Extract rBRIEF binary descriptors for given keypoints in image.

        Note that the keypoints must be extracted using the same `downscale`
        and `n_scales` parameters. Additionally, if you want to extract both
        keypoints and descriptors you should use the faster
        `detect_and_extract`.

        Parameters
        ----------
        image : 2D array
            Input image.
        keypoints : (N, 2) array
            Keypoint coordinates as ``(row, col)``.
        scales : (N, ) array
            Corresponding scales.
        orientations : (N, ) array
            Corresponding orientations in radians.

        """
        check_nD(image, 2)

        pyramid = self._build_pyramid(image)

        descriptors_list = []
        mask_list = []

        # Determine octaves from scales
        octaves = (np.log(scales) / np.log(self.downscale)).astype(np.intp)

        for octave in range(len(pyramid)):

            # Mask for all keypoints in current octave
            octave_mask = octaves == octave

            if np.sum(octave_mask) > 0:

                octave_image = np.ascontiguousarray(pyramid[octave])

                octave_keypoints = keypoints[octave_mask]
                octave_keypoints /= self.downscale ** octave
                octave_orientations = orientations[octave_mask]

                descriptors, mask = self._extract_octave(octave_image,
                                                         octave_keypoints,
                                                         octave_orientations)

                descriptors_list.append(descriptors)
                mask_list.append(mask)

        self.descriptors = np.vstack(descriptors_list).view(bool)
        self.mask_ = np.hstack(mask_list)

    def detect_and_extract(self, image):
        """Detect oriented FAST keypoints and extract rBRIEF descriptors.

        Note that this is faster than first calling `detect` and then
        `extract`.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)

        pyramid = self._build_pyramid(image)

        keypoints_list = []
        responses_list = []
        scales_list = []
        orientations_list = []
        descriptors_list = []

        for octave in range(len(pyramid)):

            octave_image = np.ascontiguousarray(pyramid[octave])

            keypoints, orientations, responses = self._detect_octave(
                octave_image)

            if len(keypoints) == 0:
                keypoints_list.append(keypoints)
                responses_list.append(responses)
                descriptors_list.append(np.zeros((0, 256), dtype=bool))
                continue

            descriptors, mask = self._extract_octave(octave_image, keypoints,
                                                     orientations)

            scaled_keypoints = keypoints[mask] * self.downscale ** octave
            keypoints_list.append(scaled_keypoints)
            responses_list.append(responses[mask])
            orientations_list.append(orientations[mask])
            scales_list.append(self.downscale ** octave *
                               np.ones(scaled_keypoints.shape[0], dtype=np.intp))
            descriptors_list.append(descriptors)

        if len(scales_list) == 0:
            raise RuntimeError(
                "ORB found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels.")

        keypoints = np.vstack(keypoints_list)
        responses = np.hstack(responses_list)
        scales = np.hstack(scales_list)
        orientations = np.hstack(orientations_list)
        descriptors = np.vstack(descriptors_list).view(bool)

        if keypoints.shape[0] < self.n_keypoints:
            self.keypoints = keypoints
            self.scales = scales
            self.orientations = orientations
            self.responses = responses
            self.descriptors = descriptors
        else:
            # Choose best n_keypoints according to Harris corner response
            best_indices = responses.argsort()[::-1][:self.n_keypoints]
            self.keypoints = keypoints[best_indices]
            self.scales = scales[best_indices]
            self.orientations = orientations[best_indices]
            self.responses = responses[best_indices]
            self.descriptors = descriptors[best_indices]
