"""
Augmenters that apply artistic image filters.

List of augmenters:

    * :class:`Cartoon`

Added in 0.4.0.

"""

from __future__ import print_function, division, absolute_import

import numpy as np
import cv2

from imgaug.imgaug import _normalize_cv2_input_arr_
from . import meta
from . import color as colorlib
from .. import dtypes as iadt
from .. import parameters as iap


def stylize_cartoon(image, blur_ksize=3, segmentation_size=1.0,
                    saturation=2.0, edge_prevalence=1.0,
                    suppress_edges=True,
                    from_colorspace=colorlib.CSPACE_RGB):
    """Convert the style of an image to a more cartoonish one.

    This function was primarily designed for images with a size of ``200``
    to ``800`` pixels. Smaller or larger images may cause issues.

    Note that the quality of the results can currently not compete with
    learned style transfer, let alone human-made images. A lack of detected
    edges or also too many detected edges are probably the most significant
    drawbacks.

    This method is loosely based on the one proposed in
    https://stackoverflow.com/a/11614479/3760780

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no

    Parameters
    ----------
    image : ndarray
        A ``(H,W,3) uint8`` image array.

    blur_ksize : int, optional
        Kernel size of the median blur filter applied initially to the input
        image. Expected to be an odd value and ``>=0``. If an even value,
        thn automatically increased to an odd one. If ``<=1``, no blur will
        be applied.

    segmentation_size : float, optional
        Size multiplier to decrease/increase the base size of the initial
        mean-shift segmentation of the image. Expected to be ``>=0``.
        Note that the base size is increased by roughly a factor of two for
        images with height and/or width ``>=400``.

    edge_prevalence : float, optional
        Multiplier for the prevalance of edges. Higher values lead to more
        edges. Note that the default value of ``1.0`` is already fairly
        conservative, so there is limit effect from lowerin it further.

    saturation : float, optional
        Multiplier for the saturation. Set to ``1.0`` to not change the
        image's saturation.

    suppress_edges : bool, optional
        Whether to run edge suppression to remove blobs containing too many
        or too few edge pixels.

    from_colorspace : str, optional
        The source colorspace. Use one of ``imgaug.augmenters.color.CSPACE_*``.
        Defaults to ``RGB``.

    Returns
    -------
    ndarray
        Image in cartoonish style.

    """
    iadt.gate_dtypes(
        image,
        allowed=["uint8"],
        disallowed=["bool",
                    "uint16", "uint32", "uint64", "uint128", "uint256",
                    "int8", "int16", "int32", "int64", "int128", "int256",
                    "float16", "float32", "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    assert image.ndim == 3 and image.shape[2] == 3, (
        "Expected to get a (H,W,C) image, got shape %s." % (image.shape,))

    blur_ksize = max(int(np.round(blur_ksize)), 1)
    segmentation_size = max(segmentation_size, 0.0)
    saturation = max(saturation, 0.0)

    is_small_image = max(image.shape[0:2]) < 400

    image = _blur_median(image, blur_ksize)
    image_seg = np.zeros_like(image)

    if is_small_image:
        spatial_window_radius = int(10 * segmentation_size)
        color_window_radius = int(20 * segmentation_size)
    else:
        spatial_window_radius = int(15 * segmentation_size)
        color_window_radius = int(40 * segmentation_size)

    if segmentation_size <= 0:
        image_seg = image
    else:
        cv2.pyrMeanShiftFiltering(_normalize_cv2_input_arr_(image),
                                  sp=spatial_window_radius,
                                  sr=color_window_radius,
                                  dst=image_seg)

    if is_small_image:
        edges_raw = _find_edges_canny(image_seg,
                                      edge_prevalence,
                                      from_colorspace)
    else:
        edges_raw = _find_edges_laplacian(image_seg,
                                          edge_prevalence,
                                          from_colorspace)

    edges = edges_raw

    edges = ((edges > 100) * 255).astype(np.uint8)

    if suppress_edges:
        # Suppress dense 3x3 blobs full of detected edges. They are visually
        # ugly.
        edges = _suppress_edge_blobs(edges, 3, 8, inverse=False)

        # Suppress spurious few-pixel edges (5x5 size with <=3 edge pixels).
        edges = _suppress_edge_blobs(edges, 5, 3, inverse=True)

    return _saturate(_blend_edges(image_seg, edges),
                     saturation,
                     from_colorspace)


# Added in 0.4.0.
def _find_edges_canny(image, edge_multiplier, from_colorspace):
    image_gray = colorlib.change_colorspace_(np.copy(image),
                                             to_colorspace=colorlib.CSPACE_GRAY,
                                             from_colorspace=from_colorspace)
    image_gray = image_gray[..., 0]
    thresh = min(int(200 * (1/edge_multiplier)), 254)
    edges = cv2.Canny(_normalize_cv2_input_arr_(image_gray), thresh, thresh)
    return edges


# Added in 0.4.0.
def _find_edges_laplacian(image, edge_multiplier, from_colorspace):
    image_gray = colorlib.change_colorspace_(np.copy(image),
                                             to_colorspace=colorlib.CSPACE_GRAY,
                                             from_colorspace=from_colorspace)
    image_gray = image_gray[..., 0]
    edges_f = cv2.Laplacian(_normalize_cv2_input_arr_(image_gray / 255.0),
                            cv2.CV_64F)
    edges_f = np.abs(edges_f)
    edges_f = edges_f ** 2
    vmax = np.percentile(edges_f, min(int(90 * (1/edge_multiplier)), 99))
    edges_f = np.clip(edges_f, 0.0, vmax) / vmax

    edges_uint8 = np.clip(np.round(edges_f * 255), 0, 255.0).astype(np.uint8)
    edges_uint8 = _blur_median(edges_uint8, 3)
    edges_uint8 = _threshold(edges_uint8, 50)
    return edges_uint8


# Added in 0.4.0.
def _blur_median(image, ksize):
    if ksize % 2 == 0:
        ksize += 1
    if ksize <= 1:
        return image
    return cv2.medianBlur(_normalize_cv2_input_arr_(image), ksize)


# Added in 0.4.0.
def _threshold(image, thresh):
    mask = (image < thresh)
    result = np.copy(image)
    result[mask] = 0
    return result


# Added in 0.4.0.
def _suppress_edge_blobs(edges, size, thresh, inverse):
    kernel = np.ones((size, size), dtype=np.float32)
    counts = cv2.filter2D(_normalize_cv2_input_arr_(edges / 255.0), -1, kernel)

    if inverse:
        mask = (counts < thresh)
    else:
        mask = (counts >= thresh)

    edges = np.copy(edges)
    edges[mask] = 0
    return edges


# Added in 0.4.0.
def _saturate(image, factor, from_colorspace):
    image = np.copy(image)
    if np.isclose(factor, 1.0, atol=1e-2):
        return image

    hsv = colorlib.change_colorspace_(image,
                                      to_colorspace=colorlib.CSPACE_HSV,
                                      from_colorspace=from_colorspace)
    sat = hsv[:, :, 1]
    sat = np.clip(sat.astype(np.int32) * factor, 0, 255).astype(np.uint8)
    hsv[:, :, 1] = sat
    image_sat = colorlib.change_colorspace_(hsv,
                                            to_colorspace=from_colorspace,
                                            from_colorspace=colorlib.CSPACE_HSV)
    return image_sat


# Added in 0.4.0.
def _blend_edges(image, image_edges):
    image_edges = 1.0 - (image_edges / 255.0)
    image_edges = np.tile(image_edges[..., np.newaxis], (1, 1, 3))
    return np.clip(
        np.round(image * image_edges),
        0.0, 255.0
    ).astype(np.uint8)


class Cartoon(meta.Augmenter):
    """Convert the style of images to a more cartoonish one.

    This augmenter was primarily designed for images with a size of ``200``
    to ``800`` pixels. Smaller or larger images may cause issues.

    Note that the quality of the results can currently not compete with
    learned style transfer, let alone human-made images. A lack of detected
    edges or also too many detected edges are probably the most significant
    drawbacks.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.artistic.stylize_cartoon`.

    Parameters
    ----------
    blur_ksize : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Median filter kernel size.
        See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

            * If ``number``: That value will be used for all images.
            * If ``tuple (a, b) of number``: A random value will be uniformly
              sampled per image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked per image from the
              ``list``.
            * If ``StochasticParameter``: The parameter will be queried once
              per batch for ``(N,)`` values, where ``N`` is the number of
              images.

    segmentation_size : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Mean-Shift segmentation size multiplier.
        See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

            * If ``number``: That value will be used for all images.
            * If ``tuple (a, b) of number``: A random value will be uniformly
              sampled per image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked per image from the
              ``list``.
            * If ``StochasticParameter``: The parameter will be queried once
              per batch for ``(N,)`` values, where ``N`` is the number of
              images.

    saturation : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Saturation multiplier.
        See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

            * If ``number``: That value will be used for all images.
            * If ``tuple (a, b) of number``: A random value will be uniformly
              sampled per image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked per image from the
              ``list``.
            * If ``StochasticParameter``: The parameter will be queried once
              per batch for ``(N,)`` values, where ``N`` is the number of
              images.

    edge_prevalence : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Multiplier for the prevalence of edges.
        See :func:`~imgaug.augmenters.artistic.stylize_cartoon` for details.

            * If ``number``: That value will be used for all images.
            * If ``tuple (a, b) of number``: A random value will be uniformly
              sampled per image from the interval ``[a, b)``.
            * If ``list``: A random value will be picked per image from the
              ``list``.
            * If ``StochasticParameter``: The parameter will be queried once
              per batch for ``(N,)`` values, where ``N`` is the number of
              images.

    from_colorspace : str, optional
        The source colorspace. Use one of ``imgaug.augmenters.color.CSPACE_*``.
        Defaults to ``RGB``.

    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.Cartoon()

    Create an example image, then apply a cartoon filter to it.

    >>> aug = iaa.Cartoon(blur_ksize=3, segmentation_size=1.0,
    >>>                   saturation=2.0, edge_prevalence=1.0)

    Create a non-stochastic cartoon augmenter that produces decent-looking
    images.

    """

    # Added in 0.4.0.
    def __init__(self, blur_ksize=(1, 5), segmentation_size=(0.8, 1.2),
                 saturation=(1.5, 2.5), edge_prevalence=(0.9, 1.1),
                 from_colorspace=colorlib.CSPACE_RGB,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Cartoon, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.blur_ksize = iap.handle_continuous_param(
            blur_ksize, "blur_ksize", value_range=(0, None),
            tuple_to_uniform=True, list_to_choice=True)
        self.segmentation_size = iap.handle_continuous_param(
            segmentation_size, "segmentation_size", value_range=(0.0, None),
            tuple_to_uniform=True, list_to_choice=True)
        self.saturation = iap.handle_continuous_param(
            saturation, "saturation", value_range=(0.0, None),
            tuple_to_uniform=True, list_to_choice=True)
        self.edge_prevalence = iap.handle_continuous_param(
            edge_prevalence, "edge_prevalence", value_range=(0.0, None),
            tuple_to_uniform=True, list_to_choice=True)
        self.from_colorspace = from_colorspace

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is not None:
            samples = self._draw_samples(batch, random_state)
            for i, image in enumerate(batch.images):
                image[...] = stylize_cartoon(
                    image,
                    blur_ksize=samples[0][i],
                    segmentation_size=samples[1][i],
                    saturation=samples[2][i],
                    edge_prevalence=samples[3][i],
                    from_colorspace=self.from_colorspace
                )
        return batch

    # Added in 0.4.0.
    def _draw_samples(self, batch, random_state):
        nb_rows = batch.nb_rows
        return (
            self.blur_ksize.draw_samples((nb_rows,), random_state=random_state),
            self.segmentation_size.draw_samples((nb_rows,),
                                                random_state=random_state),
            self.saturation.draw_samples((nb_rows,), random_state=random_state),
            self.edge_prevalence.draw_samples((nb_rows,),
                                              random_state=random_state)
        )

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.blur_ksize, self.segmentation_size, self.saturation,
                self.edge_prevalence, self.from_colorspace]
