"""
Augmenters that apply pooling operations to images.

List of augmenters:

    * :class:`AveragePooling`
    * :class:`MaxPooling`
    * :class:`MinPooling`
    * :class:`MedianPooling`

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod
import functools

import six
import numpy as np

import imgaug as ia
from . import meta
from .. import parameters as iap


def _compute_shape_after_pooling(image_shape, ksize_h, ksize_w):
    if any([axis == 0 for axis in image_shape]):
        return image_shape

    height, width = image_shape[0:2]

    if height % ksize_h > 0:
        height += ksize_h - (height % ksize_h)
    if width % ksize_w > 0:
        width += ksize_w - (width % ksize_w)

    return tuple([
        height//ksize_h,
        width//ksize_w,
    ] + list(image_shape[2:]))


@six.add_metaclass(ABCMeta)
class _AbstractPoolingBase(meta.Augmenter):
    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size, keep_size=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_AbstractPoolingBase, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.kernel_size = iap.handle_discrete_kernel_size_param(
            kernel_size,
            "kernel_size",
            value_range=(0, None),
            allow_floats=False)
        self.keep_size = keep_size

        self._resize_hm_and_sm_arrays = True

    @abstractmethod
    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        """Apply pooling method with given kernel height/width to an image."""

    def _draw_samples(self, nb_rows, random_state):
        rss = random_state.duplicate(2)
        mode = "single" if self.kernel_size[1] is None else "two"
        kernel_sizes_h = self.kernel_size[0].draw_samples(
            (nb_rows,),
            random_state=rss[0])
        if mode == "single":
            kernel_sizes_w = kernel_sizes_h
        else:
            kernel_sizes_w = self.kernel_size[1].draw_samples(
                (nb_rows,), random_state=rss[1])
        return (
            np.clip(kernel_sizes_h, 1, None),
            np.clip(kernel_sizes_w, 1, None)
        )

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None and self.keep_size:
            return batch

        samples = self._draw_samples(batch.nb_rows, random_state)
        for column in batch.columns:
            value_aug = getattr(
                self, "_augment_%s_by_samples" % (column.name,)
            )(column.value, samples)
            setattr(batch, column.attr_name, value_aug)
        return batch

    # Added in 0.4.0.
    def _augment_images_by_samples(self, images, samples):
        if not self.keep_size:
            images = list(images)

        kernel_sizes_h, kernel_sizes_w = samples

        gen = enumerate(zip(images, kernel_sizes_h, kernel_sizes_w))
        for i, (image, ksize_h, ksize_w) in gen:
            if ksize_h >= 2 or ksize_w >= 2:
                image_pooled = self._pool_image(
                    image, ksize_h, ksize_w)
                if self.keep_size:
                    image_pooled = ia.imresize_single_image(
                        image_pooled, image.shape[0:2])
                images[i] = image_pooled

        return images

    # Added in 0.4.0.
    def _augment_heatmaps_by_samples(self, heatmaps, samples):
        return self._augment_hms_and_segmaps_by_samples(heatmaps, samples,
                                                        "arr_0to1")

    # Added in 0.4.0.
    def _augment_segmentation_maps_by_samples(self, segmaps, samples):
        return self._augment_hms_and_segmaps_by_samples(segmaps, samples,
                                                        "arr")

    # Added in 0.4.0.
    def _augment_hms_and_segmaps_by_samples(self, augmentables, samples,
                                            arr_attr_name):
        if self.keep_size:
            return augmentables

        kernel_sizes_h, kernel_sizes_w = samples

        gen = enumerate(zip(augmentables, kernel_sizes_h, kernel_sizes_w))
        for i, (augmentable, ksize_h, ksize_w) in gen:
            if ksize_h >= 2 or ksize_w >= 2:
                # We could also keep the size of the HM/SM array unchanged
                # here as the library can handle HMs/SMs that are larger
                # than the image. This might be inintuitive however and
                # could lead to unnecessary performance degredation.
                if self._resize_hm_and_sm_arrays:
                    new_shape_arr = _compute_shape_after_pooling(
                        getattr(augmentable, arr_attr_name).shape,
                        ksize_h, ksize_w)
                    augmentable = augmentable.resize(new_shape_arr[0:2])

                new_shape = _compute_shape_after_pooling(
                    augmentable.shape, ksize_h, ksize_w)
                augmentable.shape = new_shape

                augmentables[i] = augmentable

        return augmentables

    # Added in 0.4.0.
    def _augment_keypoints_by_samples(self, keypoints_on_images, samples):
        if self.keep_size:
            return keypoints_on_images

        kernel_sizes_h, kernel_sizes_w = samples

        gen = enumerate(zip(keypoints_on_images, kernel_sizes_h,
                            kernel_sizes_w))
        for i, (kpsoi, ksize_h, ksize_w) in gen:
            if ksize_h >= 2 or ksize_w >= 2:
                new_shape = _compute_shape_after_pooling(
                    kpsoi.shape, ksize_h, ksize_w)

                keypoints_on_images[i] = kpsoi.on_(new_shape)

        return keypoints_on_images

    # Added in 0.4.0.
    def _augment_polygons_by_samples(self, polygons_on_images, samples):
        func = functools.partial(self._augment_keypoints_by_samples,
                                 samples=samples)
        return self._apply_to_polygons_as_keypoints(polygons_on_images, func,
                                                    recoverer=None)

    # Added in 0.4.0.
    def _augment_line_strings_by_samples(self, line_strings_on_images, samples):
        func = functools.partial(self._augment_keypoints_by_samples,
                                 samples=samples)
        return self._apply_to_cbaois_as_keypoints(line_strings_on_images, func)

    # Added in 0.4.0.
    def _augment_bounding_boxes_by_samples(self, bounding_boxes_on_images,
                                           samples):
        func = functools.partial(self._augment_keypoints_by_samples,
                                 samples=samples)
        return self._apply_to_cbaois_as_keypoints(bounding_boxes_on_images,
                                                  func)

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.kernel_size, self.keep_size]


# TODO rename kernel size parameters in all augmenters to kernel_size
# TODO add per_channel
# TODO add upscaling interpolation mode?
class AveragePooling(_AbstractPoolingBase):
    """
    Apply average pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by averaging the
    pixel values within these windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    Note that this augmenter is very similar to ``AverageBlur``.
    ``AverageBlur`` applies averaging within windows of given kernel size
    *without* striding, while ``AveragePooling`` applies striding corresponding
    to the kernel size, with optional upscaling afterwards. The upscaling
    is configured to create "pixelated"/"blocky" images by default.

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.avg_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

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
    >>> aug = iaa.AveragePooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.AveragePooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.AveragePooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.AveragePooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.AveragePooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size=(1, 5), keep_size=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(AveragePooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        return ia.avg_pool(
            image,
            (kernel_size_h, kernel_size_w)
        )


class MaxPooling(_AbstractPoolingBase):
    """
    Apply max pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by taking the
    maximum pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The maximum within each pixel window is always taken channelwise..

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.max_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

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
    >>> aug = iaa.MaxPooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.MaxPooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.MaxPooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.MaxPooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.MaxPooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size=(1, 5), keep_size=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MaxPooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        # TODO extend max_pool to support pad_mode and set it here
        #      to reflection padding
        return ia.max_pool(
            image,
            (kernel_size_h, kernel_size_w)
        )


class MinPooling(_AbstractPoolingBase):
    """
    Apply minimum pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by taking the
    minimum pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The minimum within each pixel window is always taken channelwise.

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.min_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

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
    >>> aug = iaa.MinPooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.MinPooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.MinPooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.MinPooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.MinPooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size=(1, 5), keep_size=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MinPooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        # TODO extend pool to support pad_mode and set it here
        #      to reflection padding
        return ia.min_pool(
            image,
            (kernel_size_h, kernel_size_w)
        )


class MedianPooling(_AbstractPoolingBase):
    """
    Apply median pooling to images.

    This augmenter pools images with kernel sizes ``H x W`` by taking the
    median pixel value over windows. For e.g. ``2 x 2`` this halves the image
    size. Optionally, the augmenter will automatically re-upscale the image
    to the input size (by default this is activated).

    The median within each pixel window is always taken channelwise.

    .. note::

        During heatmap or segmentation map augmentation, the respective
        arrays are not changed, only the shapes of the underlying images
        are updated. This is because imgaug can handle maps/maks that are
        larger/smaller than their corresponding image.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.median_pool`.

    Attributes
    ----------
    kernel_size : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or tuple of tuple of int or tuple of list of int or tuple of imgaug.parameters.StochasticParameter, optional
        The kernel size of the pooling operation.

        * If an int, then that value will be used for all images for both
          kernel height and width.
        * If a tuple ``(a, b)``, then a value from the discrete range
          ``[a..b]`` will be sampled per image.
        * If a list, then a random value will be sampled from that list per
          image and used for both kernel height and width.
        * If a StochasticParameter, then a value will be sampled per image
          from that parameter per image and used for both kernel height and
          width.
        * If a tuple of tuple of int given as ``((a, b), (c, d))``, then two
          values will be sampled independently from the discrete ranges
          ``[a..b]`` and ``[c..d]`` per image and used as the kernel height
          and width.
        * If a tuple of lists of int, then two values will be sampled
          independently per image, one from the first list and one from the
          second, and used as the kernel height and width.
        * If a tuple of StochasticParameter, then two values will be sampled
          indepdently per image, one from the first parameter and one from the
          second, and used as the kernel height and width.

    keep_size : bool, optional
        After pooling, the result image will usually have a different
        height/width compared to the original input image. If this
        parameter is set to True, then the pooled image will be resized
        to the input image's size, i.e. the augmenter's output shape is always
        identical to the input shape.

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
    >>> aug = iaa.MedianPooling(2)

    Create an augmenter that always pools with a kernel size of ``2 x 2``.

    >>> aug = iaa.MedianPooling(2, keep_size=False)

    Create an augmenter that always pools with a kernel size of ``2 x 2``
    and does *not* resize back to the input image size, i.e. the resulting
    images have half the resolution.

    >>> aug = iaa.MedianPooling([2, 8])

    Create an augmenter that always pools either with a kernel size
    of ``2 x 2`` or ``8 x 8``.

    >>> aug = iaa.MedianPooling((1, 7))

    Create an augmenter that always pools with a kernel size of
    ``1 x 1`` (does nothing) to ``7 x 7``. The kernel sizes are always
    symmetric.

    >>> aug = iaa.MedianPooling(((1, 7), (1, 7)))

    Create an augmenter that always pools with a kernel size of
    ``H x W`` where ``H`` and ``W`` are both sampled independently from the
    range ``[1..7]``. E.g. resulting kernel sizes could be ``3 x 7``
    or ``5 x 1``.

    """

    # TODO add floats as ksize denoting fractions of image sizes
    #      (note possible overlap with fractional kernel sizes here)
    def __init__(self, kernel_size=(1, 5), keep_size=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(MedianPooling, self).__init__(
            kernel_size=kernel_size, keep_size=keep_size,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    def _pool_image(self, image, kernel_size_h, kernel_size_w):
        # TODO extend pool to support pad_mode and set it here
        #      to reflection padding
        return ia.median_pool(
            image,
            (kernel_size_h, kernel_size_w)
        )
