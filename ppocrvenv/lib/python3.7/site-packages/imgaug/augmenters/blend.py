"""
Augmenters that blend two images with each other.

List of augmenters:

    * :class:`BlendAlpha`
    * :class:`BlendAlphaMask`
    * :class:`BlendAlphaElementwise`
    * :class:`BlendAlphaSimplexNoise`
    * :class:`BlendAlphaFrequencyNoise`
    * :class:`BlendAlphaSomeColors`
    * :class:`BlendAlphaHorizontalLinearGradient`
    * :class:`BlendAlphaVerticalLinearGradient`
    * :class:`BlendAlphaSegMapClassIds`
    * :class:`BlendAlphaBoundingBoxes`
    * :class:`BlendAlphaRegularGrid`
    * :class:`BlendAlphaCheckerboard`

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
import six
import cv2

import imgaug as ia
from imgaug.imgaug import _normalize_cv2_input_arr_
from . import meta
from .. import parameters as iap
from .. import dtypes as iadt
from .. import random as iarandom
from ..augmentables import utils as augm_utils


def _split_1d_array_to_list(arr, sizes):
    result = []
    i = 0
    for size in sizes:
        result.append(arr[i:i+size])
        i += size
    return result


def blend_alpha(image_fg, image_bg, alpha, eps=1e-2):
    """
    Blend two images using an alpha blending.

    In alpha blending, the two images are naively mixed using a multiplier.
    Let ``A`` be the foreground image and ``B`` the background image and
    ``a`` is the alpha value. Each pixel intensity is then computed as
    ``a * A_ij + (1-a) * B_ij``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested (1)
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested (1)
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested (1)
        * ``float128``: no (2)
        * ``bool``: yes; fully tested (2)

        - (1) Tests show that these dtypes work, but a conversion to
              ``float128`` happens, which only has 96 bits of size instead of
              true 128 bits and hence not twice as much resolution. It is
              possible that these dtypes result in inaccuracies, though the
              tests did not indicate that.
        - (2) Not available due to the input dtype having to be increased to
              an equivalent float dtype with two times the input resolution.
        - (3) Mapped internally to ``float16``.

    Parameters
    ----------
    image_fg : (H,W,[C]) ndarray
        Foreground image. Shape and dtype kind must match the one of the
        background image.

    image_bg : (H,W,[C]) ndarray
        Background image. Shape and dtype kind must match the one of the
        foreground image.

    alpha : number or iterable of number or ndarray
        The blending factor, between ``0.0`` and ``1.0``. Can be interpreted
        as the opacity of the foreground image. Values around ``1.0`` result
        in only the foreground image being visible. Values around ``0.0``
        result in only the background image being visible. Multiple alphas
        may be provided. In these cases, there must be exactly one alpha per
        channel in the foreground/background image. Alternatively, for
        ``(H,W,C)`` images, either one ``(H,W)`` array or an ``(H,W,C)``
        array of alphas may be provided, denoting the elementwise alpha value.

    eps : number, optional
        Controls when an alpha is to be interpreted as exactly ``1.0`` or
        exactly ``0.0``, resulting in only the foreground/background being
        visible and skipping the actual computation.

    Returns
    -------
    image_blend : (H,W,C) ndarray
        Blend of foreground and background image.

    """
    assert image_fg.shape == image_bg.shape, (
        "Expected foreground and background images to have the same shape. "
        "Got %s and %s." % (image_fg.shape, image_bg.shape))
    assert image_fg.dtype.kind == image_bg.dtype.kind, (
        "Expected foreground and background images to have the same dtype "
        "kind. Got %s and %s." % (image_fg.dtype.kind, image_bg.dtype.kind))
    # TODO switch to gate_dtypes()
    assert image_fg.dtype.name not in ["float128"], (
        "Foreground image was float128, but blend_alpha() cannot handle that "
        "dtype.")
    assert image_bg.dtype.name not in ["float128"], (
        "Background image was float128, but blend_alpha() cannot handle that "
        "dtype.")

    input_was_2d = (image_fg.ndim == 2)
    if input_was_2d:
        image_fg = image_fg[..., np.newaxis]
        image_bg = image_bg[..., np.newaxis]

    input_was_bool = False
    if image_fg.dtype.kind == "b":
        input_was_bool = True
        # use float32 instead of float16 here because it seems to be faster
        image_fg = image_fg.astype(np.float32)
        image_bg = image_bg.astype(np.float32)

    alpha = np.array(alpha, dtype=np.float64)
    if alpha.size == 1:
        pass
    else:
        if alpha.ndim == 2:
            assert alpha.shape == image_fg.shape[0:2], (
                "'alpha' given as an array must match the height and width "
                "of the foreground and background image. Got shape %s vs "
                "foreground/background shape %s." % (
                    alpha.shape, image_fg.shape))
            alpha = alpha.reshape((alpha.shape[0], alpha.shape[1], 1))
        elif alpha.ndim == 3:
            assert (
                alpha.shape == image_fg.shape
                or alpha.shape == image_fg.shape[0:2] + (1,)), (
                    "'alpha' given as an array must match the height and "
                    "width of the foreground and background image. Got "
                    "shape %s vs foreground/background shape %s." % (
                        alpha.shape, image_fg.shape))
        else:
            alpha = alpha.reshape((1, 1, -1))
        if alpha.shape[2] != image_fg.shape[2]:
            alpha = np.tile(alpha, (1, 1, image_fg.shape[2]))

    if not input_was_bool:
        if np.all(alpha >= 1.0 - eps):
            if input_was_2d:
                image_fg = image_fg[..., 0]
            return np.copy(image_fg)
        if np.all(alpha <= eps):
            if input_was_2d:
                image_bg = image_bg[..., 0]
            return np.copy(image_bg)

    # for efficiency reaons, only test one value of alpha here, even if alpha
    # is much larger
    if alpha.size > 0:
        assert 0 <= alpha.item(0) <= 1.0, (
            "Expected 'alpha' value(s) to be in the interval [0.0, 1.0]. "
            "Got min %.4f and max %.4f." % (np.min(alpha), np.max(alpha)))

    dt_images = iadt.get_minimal_dtype([image_fg, image_bg])

    # doing the below itemsize increase only for non-float images led to
    # inaccuracies for large float values
    # we also use a minimum of 4 bytes (=float32), as float32 tends to be
    # faster than float16
    isize = dt_images.itemsize * 2
    isize = max(isize, 4)
    dt_blend = np.dtype("f%d" % (isize,))

    if alpha.dtype.name != dt_blend.name:
        alpha = alpha.astype(dt_blend)
    if image_fg.dtype.name != dt_blend.name:
        image_fg = image_fg.astype(dt_blend)
    if image_bg.dtype.name != dt_blend.name:
        image_bg = image_bg.astype(dt_blend)

    # the following is equivalent to
    #     image_blend = alpha * image_fg + (1 - alpha) * image_bg
    # but supposedly faster
    image_blend = image_bg + alpha * (image_fg - image_bg)

    if input_was_bool:
        image_blend = image_blend > 0.5
    else:
        # skip clip, because alpha is expected to be in range [0.0, 1.0] and
        # both images must have same dtype dont skip round, because otherwise
        # it is very unlikely to hit the image's max possible value
        image_blend = iadt.restore_dtypes_(
            image_blend, dt_images, clip=False, round=True)

    if input_was_2d:
        return image_blend[:, :, 0]
    return image_blend


# Added in 0.4.0.
def _generate_branch_outputs(augmenter, batch, hooks, parents):
    parents_extended = parents + [augmenter]

    # Note here that the propagation hook removes columns in the batch
    # and re-adds them afterwards. So the batch should not be copied
    # after the `with` statement.
    outputs_fg = batch
    if augmenter.foreground is not None:
        outputs_fg = outputs_fg.deepcopy()
        with outputs_fg.propagation_hooks_ctx(augmenter, hooks, parents):
            if augmenter.foreground is not None:
                outputs_fg = augmenter.foreground.augment_batch_(
                    outputs_fg,
                    parents=parents_extended,
                    hooks=hooks
                )

    outputs_bg = batch
    if augmenter.background is not None:
        outputs_bg = outputs_bg.deepcopy()
        with outputs_bg.propagation_hooks_ctx(augmenter, hooks, parents):
            outputs_bg = augmenter.background.augment_batch_(
                outputs_bg,
                parents=parents_extended,
                hooks=hooks
            )

    return outputs_fg, outputs_bg


# Added in 0.4.0.
def _to_deterministic(augmenter):
    aug = augmenter.copy()
    aug.foreground = (
        aug.foreground.to_deterministic()
        if aug.foreground is not None
        else None)
    aug.background = (
        aug.background.to_deterministic()
        if aug.background is not None
        else None)
    aug.deterministic = True
    aug.random_state = augmenter.random_state.derive_rng_()
    return aug


class BlendAlpha(meta.Augmenter):
    """
    Alpha-blend two image sources using an alpha/opacity value.

    The two image sources can be imagined as branches.
    If a source is not given, it is automatically the same as the input.
    Let ``FG`` be the foreground branch and ``BG`` be the background branch.
    Then the result images are defined as ``factor * FG + (1-factor) * BG``,
    where ``factor`` is an overlay factor.

    .. note::

        It is not recommended to use ``BlendAlpha`` with augmenters
        that change the geometry of images (e.g. horizontal flips, affine
        transformations) if you *also* want to augment coordinates (e.g.
        keypoints, polygons, ...), as it is unclear which of the two
        coordinate results (foreground or background branch) should be used
        as the coordinates after augmentation.

        Currently, if ``factor >= 0.5`` (per image), the results of the
        foreground branch are used as the new coordinates, otherwise the
        results of the background branch.

    Added in 0.4.0. (Before that named `Alpha`.)

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.blend.blend_alpha`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Opacity of the results of the foreground branch. Values close to
        ``0.0`` mean that the results from the background branch (see
        parameter `background`) make up most of the final image.

            * If float, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

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
    >>> aug = iaa.BlendAlpha(0.5, iaa.Grayscale(1.0))

    Convert each image to pure grayscale and alpha-blend the result with the
    original image using an alpha of ``50%``, thereby removing about ``50%`` of
    all color. This is equivalent to ``iaa.Grayscale(0.5)``.

    >>> aug = iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0))

    Same as in the previous example, but the alpha factor is sampled uniformly
    from the interval ``[0.0, 1.0]`` once per image, thereby removing a random
    fraction of all colors. This is equivalent to
    ``iaa.Grayscale((0.0, 1.0))``.

    >>> aug = iaa.BlendAlpha(
    >>>     (0.0, 1.0),
    >>>     iaa.Affine(rotate=(-20, 20)),
    >>>     per_channel=0.5)

    First, rotate each image by a random degree sampled uniformly from the
    interval ``[-20, 20]``. Then, alpha-blend that new image with the original
    one using a random factor sampled uniformly from the interval
    ``[0.0, 1.0]``. For ``50%`` of all images, the blending happens
    channel-wise and the factor is sampled independently per channel
    (``per_channel=0.5``). As a result, e.g. the red channel may look visibly
    rotated (factor near ``1.0``), while the green and blue channels may not
    look rotated (factors near ``0.0``).

    >>> aug = iaa.BlendAlpha(
    >>>     (0.0, 1.0),
    >>>     foreground=iaa.Add(100),
    >>>     background=iaa.Multiply(0.2))

    Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
    to input images and alpha-blend the results of these branches using a
    factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
    and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
    uniformly from the interval ``[0.0, 1.0]`` per image. The resulting images
    contain a bit of ``A`` and a bit of ``B``.

    >>> aug = iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13))

    Apply median blur to each image and alpha-blend the result with the
    original image using an alpha factor of either exactly ``0.25`` or
    exactly ``0.75`` (sampled once per image).

    """

    # Added in 0.4.0.
    def __init__(self, factor=(0.0, 1.0), foreground=None, background=None,
                 per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlpha, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.factor = iap.handle_continuous_param(
            factor, "factor", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True)

        assert foreground is not None or background is not None, (
            "Expected 'foreground' and/or 'background' to not be None (i.e. "
            "at least one Augmenter), but got two None values.")
        self.foreground = meta.handle_children_list(
            foreground, self.name, "foreground", default=None)
        self.background = meta.handle_children_list(
            background, self.name, "background", default=None)

        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

        self.epsilon = 1e-2

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        batch_fg, batch_bg = _generate_branch_outputs(
            self, batch, hooks, parents)

        columns = batch.columns
        shapes = batch.get_rowwise_shapes()
        nb_images = len(shapes)
        nb_channels_max = max([shape[2] if len(shape) > 2 else 1
                               for shape in shapes])
        rngs = random_state.duplicate(2)
        per_channel = self.per_channel.draw_samples(nb_images,
                                                    random_state=rngs[0])
        alphas = self.factor.draw_samples((nb_images, nb_channels_max),
                                          random_state=rngs[1])

        for i, shape in enumerate(shapes):
            if per_channel[i] > 0.5:
                nb_channels = shape[2] if len(shape) > 2 else 1
                alphas_i = alphas[i, 0:nb_channels]
            else:
                # We catch here the case of alphas[i] being empty, which can
                # happen if all images have 0 channels.
                # In that case the alpha value doesn't matter as the image
                # contains zero values anyways.
                alphas_i = alphas[i, 0] if alphas[i].size > 0 else 0

            # compute alpha for non-image data -- average() also works with
            # scalars
            alphas_i_avg = np.average(alphas_i)
            use_fg_branch = alphas_i_avg >= 0.5

            # blend images
            if batch.images is not None:
                batch.images[i] = blend_alpha(batch_fg.images[i],
                                              batch_bg.images[i],
                                              alphas_i, eps=self.epsilon)

            # blend non-images
            # TODO Use gradual blending for heatmaps here (as for images)?
            #      Heatmaps are probably the only augmentable where this makes
            #      sense.
            for column in columns:
                if column.name != "images":
                    batch_use = (batch_fg if use_fg_branch
                                 else batch_bg)
                    column.value[i] = getattr(batch_use, column.attr_name)[i]

        return batch

    # Added in 0.4.0.
    def _to_deterministic(self):
        return _to_deterministic(self)

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.factor, self.per_channel]

    # Added in 0.4.0.
    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [lst for lst in [self.foreground, self.background]
                if lst is not None]

    # Added in 0.4.0.
    def __str__(self):
        pattern = (
            "%s("
            "factor=%s, per_channel=%s, name=%s, "
            "foreground=%s, background=%s, "
            "deterministic=%s"
            ")"
        )
        return pattern % (
            self.__class__.__name__, self.factor, self.per_channel, self.name,
            self.foreground, self.background, self.deterministic)


# tested indirectly via BlendAlphaElementwise for historic reasons
class BlendAlphaMask(meta.Augmenter):
    """
    Alpha-blend two image sources using non-binary masks generated per image.

    This augmenter queries for each image a mask generator to generate
    a ``(H,W)`` or ``(H,W,C)`` channelwise mask ``[0.0, 1.0]``, where
    ``H`` is the image height and ``W`` the width.
    The mask will then be used to alpha-blend pixel- and possibly channel-wise
    between a foreground branch of augmenters and a background branch.
    (Both branches default to the identity operation if not provided.)

    See also :class:`~imgaug.augmenters.blend.BlendAlpha`.

    .. note::

        It is not recommended to use ``BlendAlphaMask`` with augmenters
        that change the geometry of images (e.g. horizontal flips, affine
        transformations) if you *also* want to augment coordinates (e.g.
        keypoints, polygons, ...), as it is unclear which of the two
        coordinate results (foreground or background branch) should be used
        as the final output coordinates after augmentation.

        Currently, for keypoints the results of the
        foreground and background branch will be mixed. That means that for
        each coordinate the augmented result will be picked from the
        foreground or background branch based on the average alpha mask value
        at the corresponding spatial location.

        For bounding boxes, line strings and polygons, either all objects
        (on an image) of the foreground or all of the background branch will
        be used, based on the average over the whole alpha mask.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.blend.blend_alpha`.

    Parameters
    ----------
    mask_generator : IBatchwiseMaskGenerator
        A generator that will be queried per image to generate a mask.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch (i.e. identity function).
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch (i.e. identity function).
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

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
    >>> aug = iaa.BlendAlphaMask(
    >>>     iaa.InvertMaskGen(0.5, iaa.VerticalLinearGradientMaskGen()),
    >>>     iaa.Sequential([
    >>>         iaa.Clouds(),
    >>>         iaa.WithChannels([1, 2], iaa.Multiply(0.5))
    >>>     ])
    >>> )

    Create an augmenter that sometimes adds clouds at the bottom and sometimes
    at the top of the image.

    """

    # Currently the mode is only used for keypoint augmentation.
    # either or: use all keypoints from fg or all from bg branch (based
    #   on average of the whole mask).
    # pointwise: decide for each point whether to use the fg or bg
    #   branch's keypoint (based on the average mask value at the point's
    #   xy-location).
    _MODE_EITHER_OR = "either-or"
    _MODE_POINTWISE = "pointwise"
    _MODES = [_MODE_POINTWISE, _MODE_EITHER_OR]

    # Added in 0.4.0.
    def __init__(self, mask_generator,
                 foreground=None, background=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlphaMask, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.mask_generator = mask_generator

        assert foreground is not None or background is not None, (
            "Expected 'foreground' and/or 'background' to not be None (i.e. "
            "at least one Augmenter), but got two None values.")
        self.foreground = meta.handle_children_list(
            foreground, self.name, "foreground", default=None)
        self.background = meta.handle_children_list(
            background, self.name, "background", default=None)

        # this controls how keypoints and polygons are augmented
        # Non-keypoints currently uses an either-or approach.
        # Using pointwise augmentation is problematic for polygons and line
        # strings, because the order of the points may have changed (e.g.
        # from clockwise to counter-clockwise). For polygons, it is also
        # overall more likely that some child-augmenter added/deleted points
        # and we would need a polygon recoverer.
        # Overall it seems to be the better approach to use all polygons
        # from one branch or the other, which guarantuees their validity.
        # TODO decide the either-or not based on the whole average mask
        #      value but on the average mask value within the polygon's area?
        self._coord_modes = {
            "keypoints": self._MODE_POINTWISE,
            "polygons": self._MODE_EITHER_OR,
            "line_strings": self._MODE_EITHER_OR,
            "bounding_boxes": self._MODE_EITHER_OR
        }

        self.epsilon = 1e-2

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        batch_fg, batch_bg = _generate_branch_outputs(
            self, batch, hooks, parents)

        masks = self.mask_generator.draw_masks(batch, random_state)

        for i, mask in enumerate(masks):
            if batch.images is not None:
                batch.images[i] = blend_alpha(batch_fg.images[i],
                                              batch_bg.images[i],
                                              mask, eps=self.epsilon)

            if batch.heatmaps is not None:
                arr = batch.heatmaps[i].arr_0to1
                arr_height, arr_width = arr.shape[0:2]
                mask_binarized = self._binarize_mask(mask,
                                                     arr_height, arr_width)
                batch.heatmaps[i].arr_0to1 = blend_alpha(
                    batch_fg.heatmaps[i].arr_0to1,
                    batch_bg.heatmaps[i].arr_0to1,
                    mask_binarized, eps=self.epsilon)

            if batch.segmentation_maps is not None:
                arr = batch.segmentation_maps[i].arr
                arr_height, arr_width = arr.shape[0:2]
                mask_binarized = self._binarize_mask(mask,
                                                     arr_height, arr_width)
                batch.segmentation_maps[i].arr = blend_alpha(
                    batch_fg.segmentation_maps[i].arr,
                    batch_bg.segmentation_maps[i].arr,
                    mask_binarized, eps=self.epsilon)

            for augm_attr_name in ["keypoints", "bounding_boxes", "polygons",
                                   "line_strings"]:
                augm_value = getattr(batch, augm_attr_name)
                if augm_value is not None:
                    augm_value[i] = self._blend_coordinates(
                        augm_value[i],
                        getattr(batch_fg, augm_attr_name)[i],
                        getattr(batch_bg, augm_attr_name)[i],
                        mask,
                        self._coord_modes[augm_attr_name]
                    )

        return batch

    # Added in 0.4.0.
    @classmethod
    def _binarize_mask(cls, mask, arr_height, arr_width):
        # Average over channels, resize to heatmap/segmap array size
        # (+clip for cubic interpolation). We can use none-NN interpolation
        # for segmaps here as this is just the mask and not the segmap
        # array.
        mask_3d = np.atleast_3d(mask)

        # masks with zero-sized axes crash in np.average() and cannot be
        # upscaled in imresize_single_image()
        if mask.size == 0:
            mask_rs = np.zeros((arr_height, arr_width),
                               dtype=np.float32)
        else:
            mask_avg = (
                np.average(mask_3d, axis=2) if mask_3d.shape[2] > 0 else 1.0)
            mask_rs = ia.imresize_single_image(mask_avg,
                                               (arr_height, arr_width))
        mask_arr = iadt.clip_(mask_rs, 0, 1.0)
        mask_arr_binarized = (mask_arr >= 0.5)
        return mask_arr_binarized

    # Added in 0.4.0.
    @classmethod
    def _blend_coordinates(cls, cbaoi, cbaoi_fg, cbaoi_bg, mask_image,
                           mode):
        coords = augm_utils.convert_cbaois_to_kpsois(cbaoi)
        coords_fg = augm_utils.convert_cbaois_to_kpsois(cbaoi_fg)
        coords_bg = augm_utils.convert_cbaois_to_kpsois(cbaoi_bg)

        coords = coords.to_xy_array()
        coords_fg = coords_fg.to_xy_array()
        coords_bg = coords_bg.to_xy_array()

        assert coords.shape == coords_fg.shape == coords_bg.shape, (
            "Expected number of coordinates to not be changed by foreground "
            "or background branch in BlendAlphaMask. But input coordinates "
            "of shape %s were changed to %s (foreground) and %s "
            "(background). Make sure to not use any augmenters that affect "
            "the existence of coordinates." % (
                coords.shape, coords_fg.shape, coords_bg.shape))

        h_img, w_img = mask_image.shape[0:2]

        if mode == cls._MODE_POINTWISE:
            # Augment pointwise, i.e. check for each point and its
            # xy-location the average mask value and pick based on that
            # either the point from the foreground or background branch.
            assert len(coords_fg) == len(coords_bg), (
                "Got different numbers of coordinates before/after "
                "augmentation in BlendAlphaMask. The number of "
                "coordinates is currently not allowed to change for this "
                "augmenter. Input contained %d coordinates, foreground "
                "branch %d, backround branch %d." % (
                    len(coords), len(coords_fg), len(coords_bg)))

            coords_aug = []
            subgen = zip(coords, coords_fg, coords_bg)
            for coord, coord_fg, coord_bg in subgen:
                x_int = int(np.round(coord[0]))
                y_int = int(np.round(coord[1]))
                if 0 <= y_int < h_img and 0 <= x_int < w_img:
                    alphas_i = mask_image[y_int, x_int, ...]
                    alpha = (
                        np.average(alphas_i) if alphas_i.size > 0 else 1.0)
                    if alpha > 0.5:
                        coords_aug.append(coord_fg)
                    else:
                        coords_aug.append(coord_bg)
                else:
                    coords_aug.append((x_int, y_int))
        else:
            # Augment with an either-or approach over all points, i.e.
            # based on the average of the whole mask, either all points
            # from the foreground or all points from the background branch
            # are used.
            # Note that we ensured above that _keypoint_mode must be
            # _MODE_EITHER_OR if it wasn't _MODE_POINTWISE.
            mask_image_avg = (
                np.average(mask_image) if mask_image.size > 0 else 1.0)
            if mask_image_avg > 0.5:
                coords_aug = coords_fg
            else:
                coords_aug = coords_bg

        kpsoi_aug = ia.KeypointsOnImage.from_xy_array(
            coords_aug, shape=cbaoi.shape)
        return augm_utils.invert_convert_cbaois_to_kpsois_(cbaoi, kpsoi_aug)

    # Added in 0.4.0.
    def _to_deterministic(self):
        return _to_deterministic(self)

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.mask_generator]

    # Added in 0.4.0.
    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [lst for lst in [self.foreground, self.background]
                if lst is not None]

    # Added in 0.4.0.
    def __str__(self):
        pattern = (
            "%s("
            "mask_generator=%s, name=%s, foreground=%s, background=%s, "
            "deterministic=%s"
            ")"
        )
        return pattern % (
            self.__class__.__name__, self.mask_generator, self.name,
            self.foreground, self.background, self.deterministic)


# FIXME the output of the third example makes it look like per_channel isn't
#       working
class BlendAlphaElementwise(BlendAlphaMask):
    """
    Alpha-blend two image sources using alpha/opacity values sampled per pixel.

    This is the same as :class:`BlendAlpha`, except that the opacity factor is
    sampled once per *pixel* instead of once per *image* (or a few times per
    image, if ``BlendAlpha.per_channel`` is set to ``True``).

    See :class:`BlendAlpha` for more details.

    This class is a wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    Added in 0.4.0. (Before that named `AlphaElementwise`.)

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    factor : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Opacity of the results of the foreground branch. Values close to
        ``0.0`` mean that the results from the background branch (see
        parameter `background`) make up most of the final image.

            * If float, then that value will be used for all images.
            * If tuple ``(a, b)``, then a random value from the interval
              ``[a, b]`` will be sampled per image.
            * If a list, then a random value will be picked from that list per
              image.
            * If ``StochasticParameter``, then that parameter will be used to
              sample a value per image.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as True, otherwise as False.

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
    >>> aug = iaa.BlendAlphaElementwise(0.5, iaa.Grayscale(1.0))

    Convert each image to pure grayscale and alpha-blend the result with the
    original image using an alpha of ``50%`` for all pixels, thereby removing
    about ``50%`` of all color. This is equivalent to ``iaa.Grayscale(0.5)``.
    This is also equivalent to ``iaa.BlendAlpha(0.5, iaa.Grayscale(1.0))``, as
    the opacity has a fixed value of ``0.5`` and is hence identical for all
    pixels.

    >>> aug = iaa.BlendAlphaElementwise((0, 1.0), iaa.AddToHue(100))

    Same as in the previous example, but here with hue-shift instead
    of grayscaling and additionally the alpha factor is sampled uniformly
    from the interval ``[0.0, 1.0]`` once per pixel, thereby shifting the
    hue by a random fraction for each pixel.

    >>> aug = iaa.BlendAlphaElementwise(
    >>>     (0.0, 1.0),
    >>>     iaa.Affine(rotate=(-20, 20)),
    >>>     per_channel=0.5)

    First, rotate each image by a random degree sampled uniformly from the
    interval ``[-20, 20]``. Then, alpha-blend that new image with the original
    one using a random factor sampled uniformly from the interval
    ``[0.0, 1.0]`` per pixel. For ``50%`` of all images, the blending happens
    channel-wise and the factor is sampled independently per pixel *and*
    channel (``per_channel=0.5``). As a result, e.g. the red channel may look
    visibly rotated (factor near ``1.0``), while the green and blue channels
    may not look rotated (factors near ``0.0``).

    >>> aug = iaa.BlendAlphaElementwise(
    >>>     (0.0, 1.0),
    >>>     foreground=iaa.Add(100),
    >>>     background=iaa.Multiply(0.2))

    Apply two branches of augmenters -- ``A`` and ``B`` -- *independently*
    to input images and alpha-blend the results of these branches using a
    factor ``f``. Branch ``A`` increases image pixel intensities by ``100``
    and ``B`` multiplies the pixel intensities by ``0.2``. ``f`` is sampled
    uniformly from the interval ``[0.0, 1.0]`` per pixel. The resulting images
    contain a bit of ``A`` and a bit of ``B``.

    >>> aug = iaa.BlendAlphaElementwise([0.25, 0.75], iaa.MedianBlur(13))

    Apply median blur to each image and alpha-blend the result with the
    original image using an alpha factor of either exactly ``0.25`` or
    exactly ``0.75`` (sampled once per pixel).

    """

    # Added in 0.4.0.
    def __init__(self, factor=(0.0, 1.0), foreground=None, background=None,
                 per_channel=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        factor = iap.handle_continuous_param(
            factor, "factor", value_range=(0, 1.0), tuple_to_uniform=True,
            list_to_choice=True)
        mask_gen = StochasticParameterMaskGen(factor, per_channel)
        super(BlendAlphaElementwise, self).__init__(
            mask_gen, foreground, background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    @property
    def factor(self):
        return self.mask_generator.parameter


class BlendAlphaSimplexNoise(BlendAlphaElementwise):
    """Alpha-blend two image sources using simplex noise alpha masks.

    The alpha masks are sampled using a simplex noise method, roughly creating
    connected blobs of 1s surrounded by 0s. If nearest neighbour
    upsampling is used, these blobs can be rectangular with sharp edges.

    Added in 0.4.0. (Before that named `SimplexNoiseAlpha`.)

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaElementwise`.

    Parameters
    ----------
    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        The simplex noise is always generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If int, then that number will be used as the size for all
              iterations.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per iteration from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per iteration
              at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
              is picked. Most weight is put on ``linear``, followed by
              ``cubic``.
            * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If a string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per image.

            * If ``int``, then that number will be used as the iterations for
              all images.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per image from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per image at
              random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where ``min`` combines the noise maps by taking the (elementwise)
        minimum over all iteration's results, ``max`` the (elementwise)
        maximum and ``avg`` the (elementwise) average.

            * If ``imgaug.ALL``, then a random value will be picked per image
              from the valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a ``StochasticParameter``, then a random value will be
              sampled from that paramter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to 0.0 or 1.0).

            * If ``bool``, then a sigmoid will always (``True``) or never
              (``False``) be applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be
              applied to ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. ``5.0``) will move the saddle point towards the right, leading
        to more values close to 0.0.

            * If ``None``, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the interval ``[a, b]``.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

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
    >>> aug = iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0))

    Detect per image all edges, mark them in a black and white image and
    then alpha-blend the result with the original image using simplex noise
    masks.

    >>> aug = iaa.BlendAlphaSimplexNoise(
    >>>     iaa.EdgeDetect(1.0),
    >>>     upscale_method="nearest")

    Same as in the previous example, but using only nearest neighbour
    upscaling to scale the simplex noise masks to the final image sizes, i.e.
    no nearest linear upsampling is used. This leads to rectangles with sharp
    edges.

    >>> aug = iaa.BlendAlphaSimplexNoise(
    >>>     iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear")

    Same as in the previous example, but using only linear upscaling to
    scale the simplex noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This leads to rectangles with smooth edges.

    >>> aug = iaa.BlendAlphaSimplexNoise(
    >>>     iaa.EdgeDetect(1.0),
    >>>     sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as in the first example, but using a threshold for the sigmoid
    function that is further to the right. This is more conservative, i.e.
    the generated noise masks will be mostly black (values around ``0.0``),
    which means that most of the original images (parameter/branch
    `background`) will be kept, rather than using the results of the
    augmentation (parameter/branch `foreground`).

    """

    # Added in 0.4.0.
    def __init__(self, foreground=None, background=None, per_channel=False,
                 size_px_max=(2, 16), upscale_method=None,
                 iterations=(1, 3), aggregation_method="max",
                 sigmoid=True, sigmoid_thresh=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        upscale_method_default = iap.Choice(["nearest", "linear", "cubic"],
                                            p=[0.05, 0.6, 0.35])
        sigmoid_thresh_default = iap.Normal(0.0, 5.0)

        noise = iap.SimplexNoise(
            size_px_max=size_px_max,
            upscale_method=(upscale_method
                            if upscale_method is not None
                            else upscale_method_default)
        )

        if iterations != 1:
            noise = iap.IterativeNoiseAggregator(
                noise,
                iterations=iterations,
                aggregation_method=aggregation_method
            )

        use_sigmoid = (
            sigmoid is True
            or (ia.is_single_number(sigmoid) and sigmoid >= 0.01))
        if use_sigmoid:
            noise = iap.Sigmoid.create_for_noise(
                noise,
                threshold=(sigmoid_thresh
                           if sigmoid_thresh is not None
                           else sigmoid_thresh_default),
                activated=sigmoid
            )

        super(BlendAlphaSimplexNoise, self).__init__(
            factor=noise, foreground=foreground, background=background,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaFrequencyNoise(BlendAlphaElementwise):
    """Alpha-blend two image sources using frequency noise masks.

    The alpha masks are sampled using frequency noise of varying scales,
    which can sometimes create large connected blobs of ``1`` s surrounded
    by ``0`` s and other times results in smaller patterns. If nearest
    neighbour upsampling is used, these blobs can be rectangular with sharp
    edges.

    Added in 0.4.0. (Before that named `FrequencyNoiseAlpha`.)

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaElementwise`.

    Parameters
    ----------
    exponent : number or tuple of number of list of number or imgaug.parameters.StochasticParameter, optional
        Exponent to use when scaling in the frequency domain.
        Sane values are in the range ``-4`` (large blobs) to ``4`` (small
        patterns). To generate cloud-like structures, use roughly ``-2``.

            * If number, then that number will be used as the exponent for all
              iterations.
            * If tuple of two numbers ``(a, b)``, then a value will be sampled
              per iteration from the interval ``[a, b]``.
            * If a list of numbers, then a value will be picked per iteration
              at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    per_channel : bool or float, optional
        Whether to use the same factor for all channels (``False``)
        or to sample a new value for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all images
        `per_channel` will be treated as ``True``, otherwise as ``False``.

    size_px_max : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        The noise is generated in a low resolution environment.
        This parameter defines the maximum size of that environment (in
        pixels). The environment is initialized at the same size as the input
        image and then downscaled, so that no side exceeds `size_px_max`
        (aspect ratio is kept).

            * If ``int``, then that number will be used as the size for all
              iterations.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per iteration from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per
              iteration at random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per iteration.

    upscale_method : None or imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        After generating the noise maps in low resolution environments, they
        have to be upscaled to the input image size. This parameter controls
        the upscaling method.

            * If ``None``, then either ``nearest`` or ``linear`` or ``cubic``
              is picked. Most weight is put on ``linear``, followed by
              ``cubic``.
            * If ``imgaug.ALL``, then either ``nearest`` or ``linear`` or
              ``area`` or ``cubic`` is picked per iteration (all same
              probability).
            * If string, then that value will be used as the method (must be
              ``nearest`` or ``linear`` or ``area`` or ``cubic``).
            * If list of string, then a random value will be picked from that
              list per iteration.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per iteration.

    iterations : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        How often to repeat the simplex noise generation process per
        image.

            * If ``int``, then that number will be used as the iterations for
              all images.
            * If tuple of two ``int`` s ``(a, b)``, then a value will be
              sampled per image from the discrete interval ``[a..b]``.
            * If a list of ``int`` s, then a value will be picked per image at
              random from that list.
            * If a ``StochasticParameter``, then a value will be sampled from
              that parameter per image.

    aggregation_method : imgaug.ALL or str or list of str or imgaug.parameters.StochasticParameter, optional
        The noise maps (from each iteration) are combined to one noise map
        using an aggregation process. This parameter defines the method used
        for that process. Valid methods are ``min``, ``max`` or ``avg``,
        where 'min' combines the noise maps by taking the (elementwise) minimum
        over all iteration's results, ``max`` the (elementwise) maximum and
        ``avg`` the (elementwise) average.

            * If ``imgaug.ALL``, then a random value will be picked per image
              from the valid ones.
            * If a string, then that value will always be used as the method.
            * If a list of string, then a random value will be picked from
              that list per image.
            * If a ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

    sigmoid : bool or number, optional
        Whether to apply a sigmoid function to the final noise maps, resulting
        in maps that have more extreme values (close to ``0.0`` or ``1.0``).

            * If ``bool``, then a sigmoid will always (``True``) or never
              (``False``) be applied.
            * If a number ``p`` with ``0<=p<=1``, then a sigmoid will be applied to
              ``p`` percent of all final noise maps.

    sigmoid_thresh : None or number or tuple of number or imgaug.parameters.StochasticParameter, optional
        Threshold of the sigmoid, when applied. Thresholds above zero
        (e.g. ``5.0``) will move the saddle point towards the right, leading to
        more values close to ``0.0``.

            * If ``None``, then ``Normal(0, 5.0)`` will be used.
            * If number, then that threshold will be used for all images.
            * If tuple of two numbers ``(a, b)``, then a random value will
              be sampled per image from the range ``[a, b]``.
            * If ``StochasticParameter``, then a random value will be sampled
              from that parameter per image.

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
    >>> aug = iaa.BlendAlphaFrequencyNoise(foreground=iaa.EdgeDetect(1.0))

    Detect per image all edges, mark them in a black and white image and
    then alpha-blend the result with the original image using frequency noise
    masks.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     upscale_method="nearest")

    Same as the first example, but using only linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This results in smooth edges.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear")

    Same as the first example, but using only linear upscaling to
    scale the frequency noise masks to the final image sizes, i.e. no nearest
    neighbour upsampling is used. This results in smooth edges.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     upscale_method="linear",
    >>>     exponent=-2,
    >>>     sigmoid=False)

    Same as in the previous example, but with the exponent set to a constant
    ``-2`` and the sigmoid deactivated, resulting in cloud-like patterns
    without sharp edges.

    >>> aug = iaa.BlendAlphaFrequencyNoise(
    >>>     foreground=iaa.EdgeDetect(1.0),
    >>>     sigmoid_thresh=iap.Normal(10.0, 5.0))

    Same as the first example, but using a threshold for the sigmoid function
    that is further to the right. This is more conservative, i.e. the generated
    noise masks will be mostly black (values around ``0.0``), which means that
    most of the original images (parameter/branch `background`) will be kept,
    rather than using the results of the augmentation (parameter/branch
    `foreground`).

    """

    # Added in 0.4.0.
    def __init__(self, exponent=(-4, 4), foreground=None, background=None,
                 per_channel=False, size_px_max=(4, 16), upscale_method=None,
                 iterations=(1, 3), aggregation_method=["avg", "max"],
                 sigmoid=0.5, sigmoid_thresh=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        upscale_method_default = iap.Choice(["nearest", "linear", "cubic"],
                                            p=[0.05, 0.6, 0.35])
        sigmoid_thresh_default = iap.Normal(0.0, 5.0)

        noise = iap.FrequencyNoise(
            exponent=exponent,
            size_px_max=size_px_max,
            upscale_method=(upscale_method
                            if upscale_method is not None
                            else upscale_method_default)
        )

        if iterations != 1:
            noise = iap.IterativeNoiseAggregator(
                noise,
                iterations=iterations,
                aggregation_method=aggregation_method
            )

        use_sigmoid = (
            sigmoid is True
            or (ia.is_single_number(sigmoid) and sigmoid >= 0.01))
        if use_sigmoid:
            noise = iap.Sigmoid.create_for_noise(
                noise,
                threshold=(sigmoid_thresh
                           if sigmoid_thresh is not None
                           else sigmoid_thresh_default),
                activated=sigmoid
            )

        super(BlendAlphaFrequencyNoise, self).__init__(
            factor=noise, foreground=foreground, background=background,
            per_channel=per_channel,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaSomeColors(BlendAlphaMask):
    """Blend images from two branches using colorwise masks.

    This class generates masks that "mark" a few colors and replace the
    pixels within these colors with the results of the foreground branch.
    The remaining pixels are replaced with the results of the background
    branch (usually the identity function). That allows to e.g. selectively
    grayscale a few colors, while keeping other colors unchanged.

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

    .. note::

        The underlying mask generator will produce an ``AssertionError`` for
        batches that contain no images.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    nb_bins : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

    smoothness : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

    rotation_deg : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

    from_colorspace : str, optional
        See :class:`~imgaug.augmenters.blend.SomeColorsMaskGen`.

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
    >>> aug = iaa.BlendAlphaSomeColors(iaa.Grayscale(1.0))

    Create an augmenter that turns randomly removes some colors in images by
    grayscaling them.

    >>> aug = iaa.BlendAlphaSomeColors(iaa.TotalDropout(1.0))

    Create an augmenter that removes some colors in images by replacing them
    with black pixels.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.MultiplySaturation(0.5), iaa.MultiplySaturation(1.5))

    Create an augmenter that desaturates some colors and increases the
    saturation of the remaining ones.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.AveragePooling(7), alpha=[0.0, 1.0], smoothness=0.0)

    Create an augmenter that applies average pooling to some colors.
    Each color tune is either selected (alpha of ``1.0``) or not
    selected (``0.0``). There is no gradual change between similar colors.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.AveragePooling(7), nb_bins=2, smoothness=0.0)

    Create an augmenter that applies average pooling to some colors.
    Choose on average half of all colors in images for the blending operation.

    >>> aug = iaa.BlendAlphaSomeColors(
    >>>     iaa.AveragePooling(7), from_colorspace="BGR")

    Create an augmenter that applies average pooling to some colors with
    input images being in BGR colorspace.

    """

    # Added in 0.4.0.
    def __init__(self, foreground=None, background=None,
                 nb_bins=(5, 15), smoothness=(0.1, 0.3),
                 alpha=[0.0, 1.0], rotation_deg=(0, 360),
                 from_colorspace="RGB",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(BlendAlphaSomeColors, self).__init__(
            SomeColorsMaskGen(
                nb_bins=nb_bins,
                smoothness=smoothness,
                alpha=alpha,
                rotation_deg=rotation_deg,
                from_colorspace=from_colorspace
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaHorizontalLinearGradient(BlendAlphaMask):
    """Blend images from two branches along a horizontal linear gradient.

    This class generates a horizontal linear gradient mask (i.e. usually a
    mask with low values on the left and high values on the right) and
    alphas-blends between foreground and background branch using that
    mask.

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    min_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

    max_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

    start_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

    end_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`.

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
    >>> aug = iaa.BlendAlphaHorizontalLinearGradient(iaa.AddToHue((-100, 100)))

    Create an augmenter that randomizes the hue towards the right of the
    image.

    >>> aug = iaa.BlendAlphaHorizontalLinearGradient(
    >>>     iaa.TotalDropout(1.0),
    >>>     min_value=0.2, max_value=0.8)

    Create an augmenter that replaces pixels towards the right with darker
    and darker values. However it always keeps at least
    20% (``1.0 - max_value``) of the original pixel value on the far right
    and always replaces at least 20% on the far left (``min_value=0.2``).

    >>> aug = iaa.BlendAlphaHorizontalLinearGradient(
    >>>     iaa.AveragePooling(11),
    >>>     start_at=(0.0, 1.0), end_at=(0.0, 1.0))

    Create an augmenter that blends with an average-pooled image according
    to a horizontal gradient that starts at a random x-coordinate and reaches
    its maximum at another random x-coordinate. Due to that randomness,
    the gradient may increase towards the left or right.

    """

    # Added in 0.4.0.
    def __init__(self, foreground=None, background=None,
                 min_value=(0.0, 0.2), max_value=(0.8, 1.0),
                 start_at=(0.0, 0.2), end_at=(0.8, 1.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlphaHorizontalLinearGradient, self).__init__(
            HorizontalLinearGradientMaskGen(
                min_value=min_value,
                max_value=max_value,
                start_at=start_at,
                end_at=end_at
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaVerticalLinearGradient(BlendAlphaMask):
    """Blend images from two branches along a vertical linear gradient.

    This class generates a vertical linear gradient mask (i.e. usually a
    mask with low values on the left and high values on the right) and
    alphas-blends between foreground and background branch using that
    mask.

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    min_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

    max_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

    start_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

    end_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.VerticalLinearGradientMaskGen`.

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
    >>> aug = iaa.BlendAlphaVerticalLinearGradient(iaa.AddToHue((-100, 100)))

    Create an augmenter that randomizes the hue towards the bottom of the
    image.

    >>> aug = iaa.BlendAlphaVerticalLinearGradient(
    >>>     iaa.TotalDropout(1.0),
    >>>     min_value=0.2, max_value=0.8)

    Create an augmenter that replaces pixels towards the bottom with darker
    and darker values. However it always keeps at least
    20% (``1.0 - max_value``) of the original pixel value on the far bottom
    and always replaces at least 20% on the far top (``min_value=0.2``).

    >>> aug = iaa.BlendAlphaVerticalLinearGradient(
    >>>     iaa.AveragePooling(11),
    >>>     start_at=(0.0, 1.0), end_at=(0.0, 1.0))

    Create an augmenter that blends with an average-pooled image according
    to a vertical gradient that starts at a random y-coordinate and reaches
    its maximum at another random y-coordinate. Due to that randomness,
    the gradient may increase towards the bottom or top.

    >>> aug = iaa.BlendAlphaVerticalLinearGradient(
    >>>     iaa.Clouds(),
    >>>     start_at=(0.15, 0.35), end_at=0.0)

    Create an augmenter that draws clouds in roughly the top quarter of the
    image.

    """

    # Added in 0.4.0.
    def __init__(self, foreground=None, background=None,
                 min_value=(0.0, 0.2), max_value=(0.8, 1.0),
                 start_at=(0.0, 0.2), end_at=(0.8, 1.0),
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlphaVerticalLinearGradient, self).__init__(
            VerticalLinearGradientMaskGen(
                min_value=min_value,
                max_value=max_value,
                start_at=start_at,
                end_at=end_at
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaRegularGrid(BlendAlphaMask):
    """Blend images from two branches according to a regular grid.

    This class generates for each image a mask that splits the image into a
    grid-like pattern of ``H`` rows and ``W`` columns. Each cell is then
    filled with an alpha value, sampled randomly per cell.

    The difference to :class:`AlphaBlendCheckerboard` is that this class
    samples random alpha values per grid cell, while in the checkerboard the
    alpha values follow a fixed pattern.

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.RegularGridMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of rows of the checkerboard.
        See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

    nb_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of columns of the checkerboard. Analogous to `nb_rows`.
        See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Alpha value of each cell.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

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
    >>> aug = iaa.BlendAlphaRegularGrid(nb_rows=(4, 6), nb_cols=(1, 4),
    >>>                                 foreground=iaa.Multiply(0.0))

    Create an augmenter that places a ``HxW`` grid on each image, where
    ``H`` (rows) is randomly and uniformly sampled from the interval ``[4, 6]``
    and ``W`` is analogously sampled from the interval ``[1, 4]``. Roughly
    half of the cells in the grid are filled with ``0.0``, the remaining ones
    are unaltered. Which cells exactly are "dropped" is randomly decided
    per image. The resulting effect is similar to
    :class:`~imgaug.augmenters.arithmetic.CoarseDropout`.

    >>> aug = iaa.BlendAlphaRegularGrid(nb_rows=2, nb_cols=2,
    >>>                                 foreground=iaa.Multiply(0.0),
    >>>                                 background=iaa.AveragePooling(8),
    >>>                                 alpha=[0.0, 0.0, 1.0])

    Create an augmenter that always placed ``2x2`` cells on each image
    and sets about ``1/3`` of them to zero (foreground branch) and
    the remaining ``2/3`` to a pixelated version (background branch).

    """

    # Added in 0.4.0.
    def __init__(self, nb_rows, nb_cols,
                 foreground=None, background=None,
                 alpha=[0.0, 1.0],
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        # pylint: disable=dangerous-default-value
        super(BlendAlphaRegularGrid, self).__init__(
            RegularGridMaskGen(
                nb_rows=nb_rows,
                nb_cols=nb_cols,
                alpha=alpha
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaCheckerboard(BlendAlphaMask):
    """Blend images from two branches according to a checkerboard pattern.

    This class generates for each image a mask following a checkboard layout of
    ``H`` rows and ``W`` columns. Each cell is then filled with either
    ``1.0`` or ``0.0``. The cell at the top-left is always ``1.0``. Its right
    and bottom neighbour cells are ``0.0``. The 4-neighbours of any cell always
    have a value opposite to the cell's value (``0.0`` vs. ``1.0``).

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.CheckerboardMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of rows of the checkerboard.
        See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

    nb_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of columns of the checkerboard. Analogous to `nb_rows`.
        See :class:`~imgaug.augmenters.blend.CheckerboardMaskGen` for details.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

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
    >>> aug = iaa.BlendAlphaCheckerboard(nb_rows=2, nb_cols=(1, 4),
    >>>                                  foreground=iaa.AddToHue((-100, 100)))

    Create an augmenter that places a ``HxW`` grid on each image, where
    ``H`` (rows) is always ``2`` and ``W`` is randomly and uniformly sampled
    from the interval ``[1, 4]``. For half of the cells in the grid the hue
    is randomly modified, the other half of the cells is unaltered.

    """

    # Added in 0.4.0.
    def __init__(self, nb_rows, nb_cols,
                 foreground=None, background=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlphaCheckerboard, self).__init__(
            CheckerboardMaskGen(
                nb_rows=nb_rows,
                nb_cols=nb_cols
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaSegMapClassIds(BlendAlphaMask):
    """Blend images from two branches based on segmentation map ids.

    This class generates masks that are ``1.0`` at pixel locations covered
    by specific classes in segmentation maps.

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.SegMapClassIdsMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    .. note::

        Segmentation maps can have multiple channels. If that is the case
        then for each position ``(x, y)`` it is sufficient that any class id
        in any channel matches one of the desired class ids.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        segmentation maps in a batch.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    class_ids : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        See :class:`~imgaug.augmenters.blend.SegMapClassIdsMaskGen`.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    nb_sample_classes : None or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.SegMapClassIdsMaskGen`.

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
    >>> aug = iaa.BlendAlphaSegMapClassIds(
    >>>     [1, 3],
    >>>     foreground=iaa.AddToHue((-100, 100)))

    Create an augmenter that randomizes the hue wherever the segmentation maps
    contain the classes ``1`` or ``3``.

    >>> aug = iaa.BlendAlphaSegMapClassIds(
    >>>     [1, 2, 3, 4],
    >>>     nb_sample_classes=2,
    >>>     foreground=iaa.GaussianBlur(3.0))

    Create an augmenter that randomly picks ``2`` classes from the
    list ``[1, 2, 3, 4]`` and blurs the image content wherever these classes
    appear in the segmentation map. Note that as the sampling of class ids
    happens *with replacement*, it is not guaranteed to sample two *unique*
    class ids.

    >>> aug = iaa.Sometimes(0.2,
    >>>     iaa.BlendAlphaSegMapClassIds(
    >>>         2,
    >>>         background=iaa.TotalDropout(1.0)))

    Create an augmenter that zeros for roughly every fifth image all
    image pixels that do *not* belong to class id ``2`` (note that the
    `background` branch was used, not the `foreground` branch).
    Example use case: Human body landmark detection where both the
    landmarks/keypoints and the body segmentation map are known. Train the
    model to detect landmarks and sometimes remove all non-body information
    to force the model to become more independent of the background.

    """

    # Added in 0.4.0.
    def __init__(self,
                 class_ids,
                 foreground=None, background=None,
                 nb_sample_classes=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlphaSegMapClassIds, self).__init__(
            SegMapClassIdsMaskGen(
                class_ids=class_ids,
                nb_sample_classes=nb_sample_classes
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class BlendAlphaBoundingBoxes(BlendAlphaMask):
    """Blend images from two branches based on areas enclosed in bounding boxes.

    This class generates masks that are ``1.0`` within bounding boxes of given
    labels. A mask pixel will be set to ``1.0`` if *at least* one bounding box
    covers the area and has one of the requested labels.

    This class is a thin wrapper around
    :class:`~imgaug.augmenters.blend.BlendAlphaMask` together with
    :class:`~imgaug.augmenters.blend.BoundingBoxesMaskGen`.

    .. note::

        Avoid using augmenters as children that affect pixel locations (e.g.
        horizontal flips). See
        :class:`~imgaug.augmenters.blend.BlendAlphaMask` for details.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        bounding boxes in a batch.

    Added in 0.4.0.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Parameters
    ----------
    labels : None or str or list of str or imgaug.parameters.StochasticParameter
        See :class:`~imgaug.augmenters.blend.BoundingBoxesMaskGen`.

    foreground : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the foreground branch.
        High alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the foreground branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    background : None or imgaug.augmenters.meta.Augmenter or iterable of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) that make up the background branch.
        Low alpha values will show this branch's results.

            * If ``None``, then the input images will be reused as the output
              of the background branch.
            * If ``Augmenter``, then that augmenter will be used as the branch.
            * If iterable of ``Augmenter``, then that iterable will be
              converted into a ``Sequential`` and used as the augmenter.

    nb_sample_labels : None or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        See :class:`~imgaug.augmenters.blend.BoundingBoxesMaskGen`.

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
    >>> aug = iaa.BlendAlphaBoundingBoxes("person",
    >>>                                   foreground=iaa.Grayscale(1.0))

    Create an augmenter that removes color within bounding boxes having the
    label ``person``.

    >>> aug = iaa.BlendAlphaBoundingBoxes(["person", "car"],
    >>>                                   foreground=iaa.AddToHue((-255, 255)))

    Create an augmenter that randomizes the hue within bounding boxes that
    have the label ``person`` or ``car``.

    >>> aug = iaa.BlendAlphaBoundingBoxes(["person", "car"],
    >>>                                   foreground=iaa.AddToHue((-255, 255)),
    >>>                                   nb_sample_labels=1)

    Create an augmenter that randomizes the hue within bounding boxes that
    have either the label ``person`` or ``car``. Only one label is picked per
    image. Note that the sampling happens with replacement, so if
    ``nb_sample_classes`` would be ``>1``, it could still lead to only one
    *unique* label being sampled.

    >>> aug = iaa.BlendAlphaBoundingBoxes(None,
    >>>                                   background=iaa.Multiply(0.0))

    Create an augmenter that zeros all pixels (``Multiply(0.0)``)
    that are *not* (``background`` branch) within bounding boxes of
    *any* (``None``) label. In other words, all pixels outside of bounding
    boxes become black.
    Note that we don't use ``TotalDropout`` here, because by default it will
    also remove all coordinate-based augmentables, which will break the
    blending of such inputs.

    """

    # Added in 0.4.0.
    def __init__(self,
                 labels,
                 foreground=None, background=None,
                 nb_sample_labels=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(BlendAlphaBoundingBoxes, self).__init__(
            BoundingBoxesMaskGen(
                labels=labels,
                nb_sample_labels=nb_sample_labels
            ),
            foreground=foreground,
            background=background,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


@six.add_metaclass(ABCMeta)
class IBatchwiseMaskGenerator(object):
    """Interface for classes generating masks for batches.

    Child classes are supposed to receive a batch and generate an iterable
    of masks, one per row (i.e. image), matching the row shape (i.e. image
    shape). This is used in :class:`~imgaug.augmenters.blend.BlendAlphaMask`.

    Added in 0.4.0.

    """

    # Added in 0.4.0.
    def draw_masks(self, batch, random_state=None):
        """Generate a mask with given shape.

        Parameters
        ----------
        batch : imgaug.augmentables.batches._BatchInAugmentation
            Shape of the mask to sample.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            A seed or random number generator to use during the sampling
            process. If ``None``, the global RNG will be used.
            See also :func:`~imgaug.augmenters.meta.Augmenter.__init__`
            for a similar parameter with more details.

        Returns
        -------
        iterable of ndarray
            Masks, one per row in the batch.
            Each mask must be a ``float32`` array in interval ``[0.0, 1.0]``.
            It must either have the same shape as the row (i.e. the image)
            or shape ``(H, W)`` if all channels are supposed to have the
            same mask.

        """


class StochasticParameterMaskGen(IBatchwiseMaskGenerator):
    """Mask generator that queries stochastic parameters for mask values.

    This class receives batches for which to generate masks, iterates over
    the batch rows (i.e. images) and generates one mask per row.
    For a row with shape ``(H, W, C)`` (= image shape), it generates
    either a ``(H, W)`` mask (if ``per_channel`` is false-like) or a
    ``(H, W, C)`` mask (if ``per_channel`` is true-like).
    The ``per_channel`` is sampled per batch for each row/image.

    Added in 0.4.0.

    Parameters
    ----------
    parameter : imgaug.parameters.StochasticParameter
        Stochastic parameter to draw mask samples from.
        Expected to return values in interval ``[0.0, 1.0]`` (not all
        stochastic parameters do that) and must be able to handle sampling
        shapes ``(H, W)`` and ``(H, W, C)`` (all stochastic parameters should
        do that).

    per_channel : bool or float or imgaug.parameters.StochasticParameter, optional
        Whether to use the same mask for all channels (``False``)
        or to sample a new mask for each channel (``True``).
        If this value is a float ``p``, then for ``p`` percent of all rows
        (i.e. images) `per_channel` will be treated as ``True``, otherwise
        as ``False``.

    """

    # Added in 0.4.0.
    def __init__(self, parameter, per_channel):
        super(StochasticParameterMaskGen, self).__init__()
        self.parameter = parameter
        self.per_channel = iap.handle_probability_param(per_channel,
                                                        "per_channel")

    # Added in 0.4.0.
    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        """
        shapes = batch.get_rowwise_shapes()
        random_state = iarandom.RNG(random_state)
        per_channel = self.per_channel.draw_samples((len(shapes),),
                                                    random_state=random_state)

        return [self._draw_mask(shape, random_state, per_channel_i)
                for shape, per_channel_i
                in zip(shapes, per_channel)]

    # Added in 0.4.0.
    def _draw_mask(self, shape, random_state, per_channel):
        if len(shape) == 2 or per_channel >= 0.5:
            mask = self.parameter.draw_samples(shape,
                                               random_state=random_state)
        else:
            # TODO When this was wrongly sampled directly as (H,W,C) no
            #      test for AlphaElementwise ended up failing. That should not
            #      happen.

            # We are guarantueed here to have (H, W, C) as shape (H, W) is
            # handled by the above block.
            # As the mask is not channelwise, we will just return (H, W)
            # instead of (H, W, C).
            mask = self.parameter.draw_samples(shape[0:2],
                                               random_state=random_state)

        # mask has no elements if height or width in shape is 0
        if mask.size > 0:
            assert 0 <= mask.item(0) <= 1.0, (
                "Expected 'parameter' samples to be in the interval "
                "[0.0, 1.0]. Got min %.4f and max %.4f." % (
                    np.min(mask), np.max(mask),))

        return mask


class SomeColorsMaskGen(IBatchwiseMaskGenerator):
    """Generator that produces masks based on some similar colors in images.

    This class receives batches for which to generate masks, iterates over
    the batch rows (i.e. images) and generates one mask per row.
    The mask contains high alpha values for some colors, while other colors
    get low mask values. Which colors are chosen is random. How wide or
    narrow the selection is (e.g. very specific blue tone or all blue-ish
    colors) is determined by the hyperparameters.

    The color selection method performs roughly the following steps:

      1. Split the full color range of the hue in ``HSV`` into ``nb_bins``
         bins (i.e. ``256/nb_bins`` different possible hue tones).
      2. Shift the bins by ``rotation_deg`` degrees. (This way, the ``0th``
         bin does not always start at exactly ``0deg`` of hue.)
      3. Sample ``alpha`` values for each bin.
      4. Repeat the ``nb_bins`` bins until there are ``256`` bins.
      5. Smoothen the alpha values of neighbouring bins using a gaussian
         kernel. The kernel's ``sigma`` is derived from ``smoothness``.
      6. Associate all hue values in the image with the corresponding bin's
         alpha value. This results in the alpha mask.

    .. note::

        This mask generator will produce an ``AssertionError`` for batches
        that contain no images.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.color.change_colorspaces_`.

    Parameters
    ----------
    nb_bins : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of bins. For ``B`` bins, each bin denotes roughly ``360/B``
        degrees of colors in the hue channel. Lower values lead to a coarser
        selection of colors. Expected value range is ``[2, 256]``.

            * If ``int``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    smoothness : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Strength of the 1D gaussian kernel applied to the sampled binwise
        alpha values. Larger values will lead to more similar grayscaling of
        neighbouring colors. Expected value range is ``[0.0, 1.0]``.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Parameter to sample binwise alpha blending factors from. Expected
        value range is ``[0.0, 1.0]``.  Note that the alpha values will be
        smoothed between neighbouring bins. Hence, it is usually a good idea
        to set this so that the probability distribution peaks are around
        ``0.0`` and ``1.0``, e.g. via a list ``[0.0, 1.0]`` or a ``Beta``
        distribution.
        It is not recommended to set this to a deterministic value, otherwise
        all bins and hence all pixels in the generated mask will have the
        same value.

            * If ``number``: Exactly that value will be used for all bins.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per bin from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per bin from that list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N*B,)`` values -- one per image and bin.

    rotation_deg : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Rotiational shift of each bin as a fraction of ``360`` degrees.
        E.g. ``0.0`` will not shift any bins, while a value of ``0.5`` will
        shift by around ``180`` degrees. This shift is mainly used so that
        the ``0th`` bin does not always start at ``0deg``. Expected value
        range is ``[-360, 360]``. This parameter can usually be kept at the
        default value.

            * If ``number``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the interval ``[a, b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    from_colorspace : str, optional
        The source colorspace (of the input images).
        See :func:`~imgaug.augmenters.color.change_colorspace_`.

    """

    # Added in 0.4.0.
    # TODO colorlib.CSPACE_RGB produces 'has no attribute' error?
    def __init__(self, nb_bins=(5, 15), smoothness=(0.1, 0.3),
                 alpha=[0.0, 1.0], rotation_deg=(0, 360),
                 from_colorspace="RGB"):
        # pylint: disable=dangerous-default-value
        super(SomeColorsMaskGen, self).__init__()

        self.nb_bins = iap.handle_discrete_param(
            nb_bins, "nb_bins", value_range=(1, 256),
            tuple_to_uniform=True, list_to_choice=True)
        self.smoothness = iap.handle_continuous_param(
            smoothness, "smoothness", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.rotation_deg = iap.handle_continuous_param(
            rotation_deg, "rotation_deg", value_range=(-360, 360),
            tuple_to_uniform=True, list_to_choice=True)
        self.from_colorspace = from_colorspace

        self.sigma_max = 10.0

    # Added in 0.4.0.
    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        """
        assert batch.images is not None, (
            "Can only generate masks for batches that contain images, but "
            "got a batch without images.")
        random_state = iarandom.RNG(random_state)
        samples = self._draw_samples(batch, random_state=random_state)

        return [self._draw_mask(image, i, samples)
                for i, image
                in enumerate(batch.images)]

    # Added in 0.4.0.
    def _draw_mask(self, image, image_idx, samples):
        return self.generate_mask(
            image,
            samples[0][image_idx],
            samples[1][image_idx] * self.sigma_max,
            samples[2][image_idx],
            self.from_colorspace)

    # Added in 0.4.0.
    def _draw_samples(self, batch, random_state):
        nb_rows = batch.nb_rows
        nb_bins = self.nb_bins.draw_samples((nb_rows,),
                                            random_state=random_state)
        smoothness = self.smoothness.draw_samples((nb_rows,),
                                                  random_state=random_state)
        alpha = self.alpha.draw_samples((np.sum(nb_bins),),
                                        random_state=random_state)
        rotation_deg = self.rotation_deg.draw_samples(
            (nb_rows,), random_state=random_state)

        nb_bins = np.clip(nb_bins, 1, 256)
        smoothness = np.clip(smoothness, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        rotation_bins = np.mod(
            np.round(rotation_deg * (256/360)).astype(np.int32),
            256)

        binwise_alphas = _split_1d_array_to_list(alpha, nb_bins)

        return binwise_alphas, smoothness, rotation_bins

    @classmethod
    def generate_mask(cls, image, binwise_alphas, sigma,
                      rotation_bins, from_colorspace):
        """Generate a colorwise alpha mask for a single image.

        Added in 0.4.0.

        Parameters
        ----------
        image : ndarray
            Image for which to generate the mask. Must have shape ``(H,W,3)``
            in colorspace `from_colorspace`.

        binwise_alphas : ndarray
            Alpha values of shape ``(B,)`` with ``B`` in ``[1, 256]``
            and values in interval ``[0.0, 1.0]``. Will be upscaled to
            256 bins by simple repetition. Each bin represents ``1/256`` th
            of the hue.

        sigma : float
            Sigma of the 1D gaussian kernel applied to the upscaled binwise
            alpha value array.

        rotation_bins : int
            By how much to rotate the 256 bin alpha array. The rotation is
            given in number of bins.

        from_colorspace : str
            Colorspace of the input image. One of
            ``imgaug.augmenters.color.CSPACE_*``.

        Returns
        -------
        ndarray
            ``float32`` mask array of shape ``(H, W)`` with values in
            ``[0.0, 1.0]``

        """
        # import has to be deferred, otherwise python 2.7 fails
        from . import color as colorlib

        image_hsv = colorlib.change_colorspace_(
            np.copy(image),
            to_colorspace=colorlib.CSPACE_HSV,
            from_colorspace=from_colorspace)

        if 0 in image_hsv.shape[0:2]:
            return np.zeros(image_hsv.shape[0:2], dtype=np.float32)

        binwise_alphas = cls._upscale_to_256_alpha_bins(binwise_alphas)
        binwise_alphas = cls._rotate_alpha_bins(binwise_alphas, rotation_bins)
        binwise_alphas_smooth = cls._smoothen_alphas(binwise_alphas, sigma)

        mask = cls._generate_pixelwise_alpha_mask(image_hsv,
                                                  binwise_alphas_smooth)

        return mask

    # Added in 0.4.0.
    @classmethod
    def _upscale_to_256_alpha_bins(cls, alphas):
        # repeat alphas bins so that B sampled bins become 256 bins
        nb_bins = len(alphas)
        nb_repeats_per_bin = int(np.ceil(256/nb_bins))
        alphas = np.repeat(alphas, (nb_repeats_per_bin,))
        alphas = alphas[0:256]
        return alphas

    # Added in 0.4.0.
    @classmethod
    def _rotate_alpha_bins(cls, alphas, rotation_bins):
        # e.g. for offset 2: abcdef -> cdefab
        # note: offset here is expected to be in [0, 256]
        if rotation_bins > 0:
            alphas = np.roll(alphas, -rotation_bins)
        return alphas

    # Added in 0.4.0.
    @classmethod
    def _smoothen_alphas(cls, alphas, sigma):
        if sigma <= 0.0+1e-2:
            return alphas

        ksize = max(int(sigma * 2.5), 3)
        ksize_y, ksize_x = (1, ksize)
        if ksize_x % 2 == 0:
            ksize_x += 1

        # we fake here cv2.BORDER_WRAP, because GaussianBlur does not
        # support that mode, i.e. we want:
        #   cdefgh|abcdefgh|abcdefg
        alphas = np.concatenate([
            alphas[-ksize_x:],
            alphas,
            alphas[:ksize_x],
        ])

        alphas = cv2.GaussianBlur(
            _normalize_cv2_input_arr_(alphas[np.newaxis, :]),
            ksize=(ksize_x, ksize_y),
            sigmaX=sigma, sigmaY=sigma,
            borderType=cv2.BORDER_REPLICATE
        )[0, :]

        # revert fake BORDER_WRAP
        alphas = alphas[ksize_x:-ksize_x]

        return alphas

    # Added in 0.4.0.
    @classmethod
    def _generate_pixelwise_alpha_mask(cls, image_hsv, hue_to_alpha):
        hue = image_hsv[:, :, 0]
        table = hue_to_alpha * 255
        table = np.clip(np.round(table), 0, 255).astype(np.uint8)
        mask = ia.apply_lut(hue, table)
        return mask.astype(np.float32) / 255.0


# Added in 0.4.0.
class _LinearGradientMaskGen(IBatchwiseMaskGenerator):
    # Added in 0.4.0.
    def __init__(self, axis, min_value=0.0, max_value=1.0,
                 start_at=0.0, end_at=1.0):
        self.axis = axis
        self.min_value = iap.handle_continuous_param(
            min_value, "min_value", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.max_value = iap.handle_continuous_param(
            max_value, "max_value", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.start_at = iap.handle_continuous_param(
            start_at, "start_at", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.end_at = iap.handle_continuous_param(
            end_at, "end_at", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)

    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        Added in 0.4.0.

        """
        random_state = iarandom.RNG(random_state)
        shapes = batch.get_rowwise_shapes()
        samples = self._draw_samples(len(shapes), random_state=random_state)

        return [self._draw_mask(shape, i, samples)
                for i, shape
                in enumerate(shapes)]

    # Added in 0.4.0.
    def _draw_mask(self, shape, image_idx, samples):
        return self.generate_mask(
            shape,
            samples[0][image_idx],
            samples[1][image_idx],
            samples[2][image_idx],
            samples[3][image_idx])

    # Added in 0.4.0.
    def _draw_samples(self, nb_rows, random_state):
        min_value = self.min_value.draw_samples((nb_rows,),
                                                random_state=random_state)
        max_value = self.max_value.draw_samples((nb_rows,),
                                                random_state=random_state)
        start_at = self.start_at.draw_samples(
            (nb_rows,), random_state=random_state)
        end_at = self.end_at.draw_samples(
            (nb_rows,), random_state=random_state)

        return min_value, max_value, start_at, end_at

    @classmethod
    @abstractmethod
    def generate_mask(cls, shape, min_value, max_value, start_at, end_at):
        """Generate a horizontal gradient mask.

        Added in 0.4.0.

        Parameters
        ----------
        shape : tuple of int
            Shape of the image. The mask will have the same height and
            width.

        min_value : number
            Minimum value of the gradient in interval ``[0.0, 1.0]``.

        max_value : number
            Maximum value of the gradient in interval ``[0.0, 1.0]``.

        start_at : number
            Position on the x-axis where the linear gradient starts, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        end_at : number
            Position on the x-axis where the linear gradient ends, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as the image.
            Values are in ``[0.0, 1.0]``.

        """

    # Added in 0.4.0.
    @classmethod
    def _generate_mask(cls, shape, axis, min_value, max_value, start_at,
                       end_at):
        height, width = shape[0:2]

        axis_size = shape[axis]
        min_value = min(max(min_value, 0.0), 1.0)
        max_value = min(max(max_value, 0.0), 1.0)

        start_at_px = min(max(int(start_at * axis_size), 0), axis_size)
        end_at_px = min(max(int(end_at * axis_size), 0), axis_size)

        inverted = False
        if end_at_px < start_at_px:
            inverted = True
            start_at_px, end_at_px = end_at_px, start_at_px

        before_grad = np.full((start_at_px,), min_value,
                              dtype=np.float32)
        grad = np.linspace(start=min_value,
                           stop=max_value,
                           num=end_at_px - start_at_px,
                           dtype=np.float32)
        after_grad = np.full((axis_size - end_at_px,), max_value,
                             dtype=np.float32)

        mask = np.concatenate((
            before_grad,
            grad,
            after_grad
        ), axis=0)

        if inverted:
            mask = 1.0 - mask

        if axis == 0:
            mask = mask[:, np.newaxis]
            mask = np.tile(mask, (1, width))
        else:
            mask = mask[np.newaxis, :]
            mask = np.tile(mask, (height, 1))

        return mask


class HorizontalLinearGradientMaskGen(_LinearGradientMaskGen):
    """Generator that produces horizontal linear gradient masks.

    This class receives batches and produces for each row (i.e. image)
    a horizontal linear gradient that matches the row's shape (i.e. image
    shape). The gradient increases linearly from a minimum value to a
    maximum value along the x-axis. The start and end points (i.e. where the
    minimum value starts to increase and where it reaches the maximum)
    may be defines as fractions of the width. E.g. for width ``100`` and
    ``start=0.25``, ``end=0.75``, the gradient would have its minimum
    in interval ``[0px, 25px]`` and its maximum in interval ``[75px, 100px]``.

    Note that this has nothing to do with a *derivative* along the x-axis.

    Added in 0.4.0.

    Parameters
    ----------
    min_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Minimum value that the mask will have up to the start point of the
        linear gradient.
        Note that `min_value` is allowed to be larger than `max_value`,
        in which case the gradient will start at the (higher) `min_value`
        and decrease towards the (lower) `max_value`.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

    max_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Maximum value that the mask will have at the end of the
        linear gradient.

        Datatypes are analogous to `min_value`.

    start_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Position on the x-axis where the linear gradient starts, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``0.0``
        is at the left of the image.
        If ``end_at < start_at`` the gradient will be inverted.

        Datatypes are analogous to `min_value`.

    end_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Position on the x-axis where the linear gradient ends, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``0.0``
        is at the right of the image.

        Datatypes are analogous to `min_value`.

    """

    # Added in 0.4.0.
    def __init__(self, min_value=(0.0, 0.2), max_value=(0.8, 1.0),
                 start_at=(0.0, 0.2), end_at=(0.8, 1.0)):
        super(HorizontalLinearGradientMaskGen, self).__init__(
            axis=1,
            min_value=min_value,
            max_value=max_value,
            start_at=start_at,
            end_at=end_at)

    @classmethod
    def generate_mask(cls, shape, min_value, max_value, start_at, end_at):
        """Generate a linear horizontal gradient mask.

        Added in 0.4.0.

        Parameters
        ----------
        shape : tuple of int
            Shape of the image. The mask will have the same height and
            width.

        min_value : number
            Minimum value of the gradient in interval ``[0.0, 1.0]``.

        max_value : number
            Maximum value of the gradient in interval ``[0.0, 1.0]``.

        start_at : number
            Position on the x-axis where the linear gradient starts, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        end_at : number
            Position on the x-axis where the linear gradient ends, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as the image.
            Values are in ``[0.0, 1.0]``.

        """
        return cls._generate_mask(
            axis=1,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            start_at=start_at,
            end_at=end_at)


class VerticalLinearGradientMaskGen(_LinearGradientMaskGen):
    """Generator that produces vertical linear gradient masks.

    See :class:`~imgaug.augmenters.blend.HorizontalLinearGradientMaskGen`
    for details.

    Added in 0.4.0.

    Parameters
    ----------
    min_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Minimum value that the mask will have up to the start point of the
        linear gradient.
        Note that `min_value` is allowed to be larger than `max_value`,
        in which case the gradient will start at the (higher) `min_value`
        and decrease towards the (lower) `max_value`.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

    max_value : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Maximum value that the mask will have at the end of the
        linear gradient.

        Datatypes are analogous to `min_value`.

    start_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Position on the y-axis where the linear gradient starts, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``0.0``
        is at the top of the image.
        If ``end_at < start_at`` the gradient will be inverted.

        Datatypes are analogous to `min_value`.

    end_at : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Position on the x-axis where the linear gradient ends, given as a
        fraction of the axis size. Interval is ``[0.0, 1.0]``, where ``1.0``
        is at the bottom of the image.

        Datatypes are analogous to `min_value`.

    """

    # Added in 0.4.0.
    def __init__(self, min_value=(0.0, 0.2), max_value=(0.8, 1.0),
                 start_at=(0.0, 0.2), end_at=(0.8, 1.0)):
        super(VerticalLinearGradientMaskGen, self).__init__(
            axis=0,
            min_value=min_value,
            max_value=max_value,
            start_at=start_at,
            end_at=end_at)

    @classmethod
    def generate_mask(cls, shape, min_value, max_value, start_at, end_at):
        """Generate a linear horizontal gradient mask.

        Added in 0.4.0.

        Parameters
        ----------
        shape : tuple of int
            Shape of the image. The mask will have the same height and
            width.

        min_value : number
            Minimum value of the gradient in interval ``[0.0, 1.0]``.

        max_value : number
            Maximum value of the gradient in interval ``[0.0, 1.0]``.

        start_at : number
            Position on the x-axis where the linear gradient starts, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        end_at : number
            Position on the x-axis where the linear gradient ends, given as
            a fraction of the axis size. Interval is ``[0.0, 1.0]``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as the image.
            Values are in ``[0.0, 1.0]``.

        """
        return cls._generate_mask(
            axis=0,
            shape=shape,
            min_value=min_value,
            max_value=max_value,
            start_at=start_at,
            end_at=end_at)


class RegularGridMaskGen(IBatchwiseMaskGenerator):
    """Generate masks following a regular grid pattern.

    This mask generator splits each image into a grid-like pattern of
    ``H`` rows and ``W`` columns. Each cell is then filled with an alpha
    value, sampled randomly per cell.

    The difference to :class:`CheckerboardMaskGen` is that this mask generator
    samples random alpha values per cell, while in the checkerboard the
    alpha values follow a fixed pattern.

    Added in 0.4.0.

    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of rows of the regular grid.

            * If ``int``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    nb_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Number of columns of the checkerboard. Analogous to `nb_rows`.

    alpha : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Alpha value of each cell.

        * If ``number``: Exactly that value will be used for all images.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
          per image from the interval ``[a, b]``.
        * If ``list``: A random value will be picked per image from that list.
        * If ``StochasticParameter``: That parameter will be queried once
          per batch for ``(N,)`` values -- one per image.

    """

    # Added in 0.4.0.
    def __init__(self, nb_rows, nb_cols, alpha=[0.0, 1.0]):
        # pylint: disable=dangerous-default-value
        self.nb_rows = iap.handle_discrete_param(
            nb_rows, "nb_rows", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True,
            allow_floats=False)
        self.nb_cols = iap.handle_discrete_param(
            nb_cols, "nb_cols", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True,
            allow_floats=False)
        self.alpha = iap.handle_continuous_param(
            alpha, "alpha", value_range=(0.0, 1.0),
            tuple_to_uniform=True, list_to_choice=True)

    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        Added in 0.4.0.

        """
        random_state = iarandom.RNG(random_state)
        shapes = batch.get_rowwise_shapes()
        nb_rows, nb_cols, alpha = self._draw_samples(len(shapes),
                                                     random_state=random_state)

        return [self.generate_mask(shape, nb_rows_i, nb_cols_i, alpha_i)
                for shape, nb_rows_i, nb_cols_i, alpha_i
                in zip(shapes, nb_rows, nb_cols, alpha)]

    # Added in 0.4.0.
    def _draw_samples(self, nb_images, random_state):
        nb_rows = self.nb_rows.draw_samples((nb_images,),
                                            random_state=random_state)
        nb_cols = self.nb_cols.draw_samples((nb_images,),
                                            random_state=random_state)
        nb_alphas_per_img = nb_rows * nb_cols
        alpha_raw = self.alpha.draw_samples(
            (np.sum(nb_alphas_per_img),),
            random_state=random_state)

        alpha = _split_1d_array_to_list(alpha_raw, nb_alphas_per_img)

        return nb_rows, nb_cols, alpha

    @classmethod
    def generate_mask(cls, shape, nb_rows, nb_cols, alphas):
        """Generate a mask following a checkerboard pattern.

        Added in 0.4.0.

        Parameters
        ----------
        shape : tuple of int
            Height and width of the output mask.

        nb_rows : int
            Number of rows of the checkerboard pattern.

        nb_cols : int
            Number of columns of the checkerboard pattern.

        alphas : ndarray
            1D or 2D array containing for each cell the alpha value, i.e.
            ``nb_rows*nb_cols`` values.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        from . import size as sizelib

        height, width = shape[0:2]
        if 0 in (height, width):
            return np.zeros((height, width), dtype=np.float32)

        nb_rows = min(max(nb_rows, 1), height)
        nb_cols = min(max(nb_cols, 1), width)

        cell_height = int(height / nb_rows)
        cell_width = int(width / nb_cols)

        # If there are more alpha values than nb_rows*nb_cols we reduce the
        # number of alpha values.
        alphas = alphas.flat[0:nb_rows*nb_cols]
        assert alphas.size == nb_rows*nb_cols, (
            "Expected `alphas` to not contain less values than "
            "`nb_rows * nb_cols` (both clipped to [1, height] and "
            "[1, width] respectively). Got %d alpha values vs %d expected "
            "values (nb_rows=%d, nb_cols=%d) for requested mask shape %s." % (
                alphas.size, nb_rows * nb_cols, nb_rows, nb_cols,
                (height, width)))
        mask = alphas.astype(np.float32).reshape((nb_rows, nb_cols))
        mask = np.repeat(mask, cell_height, axis=0)
        mask = np.repeat(mask, cell_width, axis=1)

        # if mask is too small, reflection pad it on all sides
        missing_height = height - mask.shape[0]
        missing_width = width - mask.shape[1]
        top = int(np.floor(missing_height / 2))
        bottom = int(np.ceil(missing_height / 2))
        left = int(np.floor(missing_width / 2))
        right = int(np.ceil(missing_width / 2))
        mask = sizelib.pad(mask,
                           top=top, right=right, bottom=bottom, left=left,
                           mode="reflect")

        return mask


class CheckerboardMaskGen(IBatchwiseMaskGenerator):
    """Generate masks following a checkerboard-like pattern.

    This mask generator splits each image into a regular grid of
    ``H`` rows and ``W`` columns. Each cell is then filled with either
    ``1.0`` or ``0.0``. The cell at the top-left is always ``1.0``. Its right
    and bottom neighbour cells are ``0.0``. The 4-neighbours of any cell always
    have a value opposite to the cell's value (``0.0`` vs. ``1.0``).

    Added in 0.4.0.

    Parameters
    ----------
    nb_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of the checkerboard.

            * If ``int``: Exactly that value will be used for all images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled
              per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from that
              list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(N,)`` values -- one per image.

    nb_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of columns of the checkerboard. Analogous to `nb_rows`.

    """

    def __init__(self, nb_rows, nb_cols):
        self.grid = RegularGridMaskGen(nb_rows=nb_rows,
                                       nb_cols=nb_cols,
                                       alpha=1)

    @property
    def nb_rows(self):
        """Get the number of rows of the checkerboard grid.

        Added in 0.4.0.

        Returns
        -------
        int
            The number of rows.

        """
        return self.grid.nb_rows

    @property
    def nb_cols(self):
        """Get the number of columns of the checkerboard grid.

        Added in 0.4.0.

        Returns
        -------
        int
            The number of columns.

        """
        return self.grid.nb_cols

    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        Added in 0.4.0.

        """
        # pylint: disable=protected-access
        random_state = iarandom.RNG(random_state)
        shapes = batch.get_rowwise_shapes()
        nb_rows, nb_cols, _alpha = self.grid._draw_samples(
            len(shapes), random_state=random_state)

        return [self.generate_mask(shape, nb_rows_i, nb_cols_i)
                for shape, nb_rows_i, nb_cols_i
                in zip(shapes, nb_rows, nb_cols)]

    @classmethod
    def generate_mask(cls, shape, nb_rows, nb_cols):
        """Generate a mask following a checkerboard pattern.

        Added in 0.4.0.

        Parameters
        ----------
        shape : tuple of int
            Height and width of the output mask.

        nb_rows : int
            Number of rows of the checkerboard pattern.

        nb_cols : int
            Number of columns of the checkerboard pattern.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        height, width = shape[0:2]
        if 0 in (height, width):
            return np.zeros((height, width), dtype=np.float32)
        nb_rows = min(max(nb_rows, 1), height)
        nb_cols = min(max(nb_cols, 1), width)

        alphas = np.full((nb_cols,), 1.0, dtype=np.float32)
        alphas[::2] = 0.0
        alphas = np.tile(alphas[np.newaxis, :], (nb_rows, 1))
        alphas[::2, :] = 1.0 - alphas[::2, :]

        return RegularGridMaskGen.generate_mask(shape, nb_rows, nb_cols, alphas)


class SegMapClassIdsMaskGen(IBatchwiseMaskGenerator):
    """Generator that produces masks highlighting segmentation map classes.

    This class produces for each segmentation map in a batch a mask in which
    the locations of a set of provided classes are highlighted (i.e. ``1.0``).
    The classes may be provided as a fixed list of class ids or a stochastic
    parameter from which class ids will be sampled.

    The produced masks are initially of the same height and width as the
    segmentation map arrays and later upscaled to the image height and width.

    .. note::

        Segmentation maps can have multiple channels. If that is the case
        then for each position ``(x, y)`` it is sufficient that any class id
        in any channel matches one of the desired class ids.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        segmentation maps in a batch.

    Added in 0.4.0.

    Parameters
    ----------
    class_ids : int or tuple of int or list of int or imgaug.parameters.StochasticParameter
        Segmentation map classes to mark in the produced mask.

        If `nb_sample_classes` is ``None`` then this is expected to be either
        a single ``int`` (always mark this one class id) or a ``list`` of
        ``int`` s (always mark these class ids).

        If `nb_sample_classes` is set, then this parameter will be treated
        as a stochastic parameter with the following valid types:

            * If ``int``: Exactly that class id will be used for all
              segmentation maps.
            * If ``tuple`` ``(a, b)``: ``N`` random values will be uniformly
              sampled per segmentation map from the discrete interval
              ``[a..b]`` and used as the class ids.
            * If ``list``: ``N`` random values will be picked per segmentation
              map from that list and used as the class ids.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(sum(N),)`` values.

        ``N`` denotes the number of classes to sample per segmentation
        map (derived from `nb_sample_classes`) and ``sum(N)`` denotes the
        sum of ``N`` s over all segmentation maps.

    nb_sample_classes : None or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of class ids to sample (with replacement) per segmentation map.
        As sampling happens with replacement, fewer *unique* class ids may be
        sampled.

            * If ``None``: `class_ids` is expected to be a fixed value of
              class ids to be used for all segmentation maps.
            * If ``int``: Exactly that many class ids will be sampled for all
              segmentation maps.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly
              sampled per segmentation map from the discrete interval
              ``[a..b]``.
            * If ``list`` or ``int``: A random value will be picked per
              segmentation map from that list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(B,)`` values, where ``B`` is the number of
              segmentation maps.

    """

    # Added in 0.4.0.
    def __init__(self, class_ids, nb_sample_classes=None):
        if nb_sample_classes is None:
            if ia.is_single_integer(class_ids):
                class_ids = [class_ids]
            assert isinstance(class_ids, list), (
                "Expected `class_ids` to be a single integer or a list of "
                "integers if `nb_sample_classes` is None. Got type `%s`. "
                "Set `nb_sample_classes` to e.g. an integer to enable "
                "stochastic parameters for `class_ids`." % (
                    type(class_ids).__name__,))
            self.class_ids = class_ids
            self.nb_sample_classes = None
        else:
            self.class_ids = iap.handle_discrete_param(
                class_ids, "class_ids", value_range=(0, None),
                tuple_to_uniform=True, list_to_choice=True,
                allow_floats=False)
            self.nb_sample_classes = iap.handle_discrete_param(
                nb_sample_classes, "nb_sample_classes", value_range=(0, None),
                tuple_to_uniform=True, list_to_choice=True,
                allow_floats=False)

    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        Added in 0.4.0.

        """
        assert batch.segmentation_maps is not None, (
            "Can only generate masks for batches that contain segmentation "
            "maps, but got a batch without them.")
        random_state = iarandom.RNG(random_state)
        class_ids = self._draw_samples(batch.nb_rows,
                                       random_state=random_state)

        return [self.generate_mask(segmap, class_ids_i)
                for segmap, class_ids_i
                in zip(batch.segmentation_maps, class_ids)]

    # Added in 0.4.0.
    def _draw_samples(self, nb_rows, random_state):
        nb_sample_classes = self.nb_sample_classes
        if nb_sample_classes is None:
            assert isinstance(self.class_ids, list), (
                "Expected list got %s." % (type(self.class_ids).__name__,))
            return [self.class_ids] * nb_rows

        nb_sample_classes = nb_sample_classes.draw_samples(
            (nb_rows,), random_state=random_state)
        nb_sample_classes = np.clip(nb_sample_classes, 0, None)
        class_ids_raw = self.class_ids.draw_samples(
            (np.sum(nb_sample_classes),),
            random_state=random_state)

        class_ids = _split_1d_array_to_list(class_ids_raw, nb_sample_classes)

        return class_ids

    # TODO this could be simplified to something like:
    #      segmap.keep_only_classes(class_ids).draw_mask()
    @classmethod
    def generate_mask(cls, segmap, class_ids):
        """Generate a mask of where the segmentation map has the given classes.

        Added in 0.4.0.

        Parameters
        ----------
        segmap : imgaug.augmentables.segmap.SegmentationMapsOnImage
            The segmentation map for which to generate the mask.

        class_ids : iterable of int
            IDs of the classes to set to ``1.0``.
            For an ``(x, y)`` position, it is enough that *any* channel
            at the given location to have one of these class ids to be marked
            as ``1.0``.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        mask = np.zeros(segmap.arr.shape[0:2], dtype=bool)

        for class_id in class_ids:
            # note that segmap has shape (H,W,C), so we max() along C
            mask_i = np.any(segmap.arr == class_id, axis=2)
            mask = np.logical_or(mask, mask_i)

        mask = mask.astype(np.float32)
        mask = ia.imresize_single_image(mask, segmap.shape[0:2])

        return mask


class BoundingBoxesMaskGen(IBatchwiseMaskGenerator):
    """Generator that produces masks highlighting bounding boxes.

    This class produces for each row (i.e. image + bounding boxes) in a batch
    a mask in which the inner areas of bounding box rectangles with given
    labels are marked (i.e. set to ``1.0``). The labels may be provided as a
    fixed list of strings or a stochastic parameter from which labels will be
    sampled. If no labels are provided, all bounding boxes will be marked.

    A pixel will be set to ``1.0`` if *at least* one bounding box at that
    location has one of the requested labels, even if there is *also* one
    bounding box at that location with a not requested label.

    .. note::

        This class will produce an ``AssertionError`` if there are no
        bounding boxes in a batch.

    Added in 0.4.0.

    Parameters
    ----------
    labels : None or str or list of str or imgaug.parameters.StochasticParameter
        Labels of bounding boxes to select for.

        If `nb_sample_labels` is ``None`` then this is expected to be either
        also ``None`` (select all BBs) or a single ``str`` (select BBs with
        this one label) or a ``list`` of ``str`` s (always select BBs with
        these labels).

        If `nb_sample_labels` is set, then this parameter will be treated
        as a stochastic parameter with the following valid types:

            * If ``None``: Ignore the sampling count  and always use all
              bounding boxes.
            * If ``str``: Exactly that label will be used for all
              images.
            * If ``list`` of ``str``: ``N`` random values will be picked per
              image from that list and used as the labels.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(sum(N),)`` values.

        ``N`` denotes the number of labels to sample per segmentation
        map (derived from `nb_sample_labels`) and ``sum(N)`` denotes the
        sum of ``N`` s over all images.

    nb_sample_labels : None or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of labels to sample (with replacement) per image.
        As sampling happens with replacement, fewer *unique* labels may be
        sampled.

            * If ``None``: `labels` is expected to also be ``None`` or a fixed
              value of labels to be used for all images.
            * If ``int``: Exactly that many labels will be sampled for all
              images.
            * If ``tuple`` ``(a, b)``: A random value will be uniformly
              sampled per image from the discrete interval ``[a..b]``.
            * If ``list``: A random value will be picked per image from
              that list.
            * If ``StochasticParameter``: That parameter will be queried once
              per batch for ``(B,)`` values, where ``B`` is the number of
              images.

    """

    # Added in 0.4.0.
    def __init__(self, labels=None, nb_sample_labels=None):
        if labels is None:
            self.labels = None
            self.nb_sample_labels = None
        elif nb_sample_labels is None:
            if ia.is_string(labels):
                labels = [labels]
            assert isinstance(labels, list), (
                "Expected `labels` a single string or a list of "
                "strings if `nb_sample_labels` is None. Got type `%s`. "
                "Set `nb_sample_labels` to e.g. an integer to enable "
                "stochastic parameters for `labels`." % (
                    type(labels).__name__,))
            self.labels = labels
            self.nb_sample_labels = None
        else:
            self.labels = iap.handle_categorical_string_param(labels, "labels")
            self.nb_sample_labels = iap.handle_discrete_param(
                nb_sample_labels, "nb_sample_labels", value_range=(0, None),
                tuple_to_uniform=True, list_to_choice=True,
                allow_floats=False)

    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        Added in 0.4.0.

        """
        assert batch.bounding_boxes is not None, (
            "Can only generate masks for batches that contain bounding boxes, "
            "but got a batch without them.")
        random_state = iarandom.RNG(random_state)

        if self.labels is None:
            return [self.generate_mask(bbsoi, None)
                    for bbsoi in batch.bounding_boxes]

        labels = self._draw_samples(batch.nb_rows, random_state=random_state)

        return [self.generate_mask(bbsoi, labels_i)
                for bbsoi, labels_i
                in zip(batch.bounding_boxes, labels)]

    # Added in 0.4.0.
    def _draw_samples(self, nb_rows, random_state):
        nb_sample_labels = self.nb_sample_labels
        if nb_sample_labels is None:
            assert isinstance(self.labels, list), (
                "Expected list got %s." % (type(self.labels).__name__,))
            return [self.labels] * nb_rows

        nb_sample_labels = nb_sample_labels.draw_samples(
            (nb_rows,), random_state=random_state)
        nb_sample_labels = np.clip(nb_sample_labels, 0, None)
        labels_raw = self.labels.draw_samples(
            (np.sum(nb_sample_labels),),
            random_state=random_state)

        labels = _split_1d_array_to_list(labels_raw, nb_sample_labels)

        return labels

    # TODO this could be simplified to something like
    #      bbsoi.only_labels(labels).draw_mask()
    @classmethod
    def generate_mask(cls, bbsoi, labels):
        """Generate a mask of the areas of bounding boxes with given labels.

        Added in 0.4.0.

        Parameters
        ----------
        bbsoi : imgaug.augmentables.bbs.BoundingBoxesOnImage
            The bounding boxes for which to generate the mask.

        labels : None or iterable of str
            Labels of the bounding boxes to set to ``1.0``.
            For an ``(x, y)`` position, it is enough that *any* bounding box
            at the given location has one of the labels.
            If this is ``None``, all bounding boxes will be marked.

        Returns
        -------
        ndarray
            ``float32`` mask array with same height and width as
            ``segmap.shape``. Values are in ``[0.0, 1.0]``.

        """
        labels = set(labels) if labels is not None else None
        height, width = bbsoi.shape[0:2]
        mask = np.zeros((height, width), dtype=np.float32)

        for bb in bbsoi:
            if labels is None or bb.label in labels:
                x1 = min(max(int(bb.x1), 0), width)
                y1 = min(max(int(bb.y1), 0), height)
                x2 = min(max(int(bb.x2), 0), width)
                y2 = min(max(int(bb.y2), 0), height)
                if x1 < x2 and y1 < y2:
                    mask[y1:y2, x1:x2] = 1.0

        return mask


class InvertMaskGen(IBatchwiseMaskGenerator):
    """Generator that inverts the outputs of other mask generators.

    This class receives batches and calls for each row (i.e. image)
    a child mask generator to produce a mask. That mask is then inverted
    for ``p%`` of all rows, i.e. converted to ``1.0 - mask``.

    Added in 0.4.0.

    Parameters
    ----------
    p : bool or float or imgaug.parameters.StochasticParameter, optional
        Probability of inverting each mask produced by the other mask
        generator.

    child : IBatchwiseMaskGenerator
        The other mask generator to invert.

    """

    # Added in 0.4.0.
    def __init__(self, p, child):
        self.p = iap.handle_probability_param(p, "p")
        self.child = child

    def draw_masks(self, batch, random_state=None):
        """
        See :func:`~imgaug.augmenters.blend.IBatchwiseMaskGenerator.draw_masks`.

        Added in 0.4.0.

        """
        random_state = iarandom.RNG(random_state)
        masks = self.child.draw_masks(batch, random_state=random_state)
        p = self.p.draw_samples(len(masks), random_state=random_state)
        for mask, p_i in zip(masks, p):
            if p_i >= 0.5:
                mask[...] = 1.0 - mask
        return masks


@ia.deprecated(alt_func="Alpha",
               comment="Alpha is deprecated. "
                       "Use BlendAlpha instead. "
                       "The order of parameters is the same. "
                       "Parameter 'first' was renamed to 'foreground'. "
                       "Parameter 'second' was renamed to 'background'.")
def Alpha(factor=0, first=None, second=None, per_channel=False,
          seed=None, name=None,
          random_state="deprecated", deterministic="deprecated"):
    """See :class:`BlendAlpha`.

    Deprecated since 0.4.0.

    """
    # pylint: disable=invalid-name
    return BlendAlpha(
        factor=factor,
        foreground=first,
        background=second,
        per_channel=per_channel,
        seed=seed, name=name,
        random_state=random_state, deterministic=deterministic)


@ia.deprecated(alt_func="AlphaElementwise",
               comment="AlphaElementwise is deprecated. "
                       "Use BlendAlphaElementwise instead. "
                       "The order of parameters is the same. "
                       "Parameter 'first' was renamed to 'foreground'. "
                       "Parameter 'second' was renamed to 'background'.")
def AlphaElementwise(factor=0, first=None, second=None, per_channel=False,
                     seed=None, name=None,
                     random_state="deprecated", deterministic="deprecated"):
    """See :class:`BlendAlphaElementwise`.

    Deprecated since 0.4.0.

    """
    # pylint: disable=invalid-name
    return BlendAlphaElementwise(
        factor=factor,
        foreground=first,
        background=second,
        per_channel=per_channel,
        seed=seed, name=name,
        random_state=random_state, deterministic=deterministic)


@ia.deprecated(alt_func="BlendAlphaSimplexNoise",
               comment="SimplexNoiseAlpha is deprecated. "
                       "Use BlendAlphaSimplexNoise instead. "
                       "The order of parameters is the same. "
                       "Parameter 'first' was renamed to 'foreground'. "
                       "Parameter 'second' was renamed to 'background'.")
def SimplexNoiseAlpha(first=None, second=None, per_channel=False,
                      size_px_max=(2, 16), upscale_method=None,
                      iterations=(1, 3), aggregation_method="max",
                      sigmoid=True, sigmoid_thresh=None,
                      seed=None, name=None,
                      random_state="deprecated", deterministic="deprecated"):
    """See :class:`BlendAlphaSimplexNoise`.

    Deprecated since 0.4.0.

    """
    # pylint: disable=invalid-name
    return BlendAlphaSimplexNoise(
        foreground=first,
        background=second,
        per_channel=per_channel,
        size_px_max=size_px_max,
        upscale_method=upscale_method,
        iterations=iterations,
        aggregation_method=aggregation_method,
        sigmoid=sigmoid,
        sigmoid_thresh=sigmoid_thresh,
        seed=seed, name=name,
        random_state=random_state, deterministic=deterministic)


@ia.deprecated(alt_func="BlendAlphaFrequencyNoise",
               comment="FrequencyNoiseAlpha is deprecated. "
                       "Use BlendAlphaFrequencyNoise instead. "
                       "The order of parameters is the same. "
                       "Parameter 'first' was renamed to 'foreground'. "
                       "Parameter 'second' was renamed to 'background'.")
def FrequencyNoiseAlpha(exponent=(-4, 4), first=None, second=None,
                        per_channel=False, size_px_max=(4, 16),
                        upscale_method=None,
                        iterations=(1, 3), aggregation_method=["avg", "max"],
                        sigmoid=0.5, sigmoid_thresh=None,
                        seed=None, name=None,
                        random_state="deprecated", deterministic="deprecated"):
    """See :class:`BlendAlphaFrequencyNoise`.

    Deprecated since 0.4.0.

    """
    # pylint: disable=invalid-name, dangerous-default-value
    return BlendAlphaFrequencyNoise(
        exponent=exponent,
        foreground=first,
        background=second,
        per_channel=per_channel,
        size_px_max=size_px_max,
        upscale_method=upscale_method,
        iterations=iterations,
        aggregation_method=aggregation_method,
        sigmoid=sigmoid,
        sigmoid_thresh=sigmoid_thresh,
        seed=seed, name=name,
        random_state=random_state, deterministic=deterministic)
