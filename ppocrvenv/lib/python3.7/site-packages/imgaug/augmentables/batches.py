"""Classes representing batches of normalized or unnormalized data."""
from __future__ import print_function, division, absolute_import

import collections

import numpy as np

from .. import imgaug as ia
from . import normalization as nlib
from . import utils

DEFAULT = "DEFAULT"

_AUGMENTABLE_NAMES = [
    "images", "heatmaps", "segmentation_maps", "keypoints",
    "bounding_boxes", "polygons", "line_strings"]

_AugmentableColumn = collections.namedtuple(
    "_AugmentableColumn",
    ["name", "value", "attr_name"])


def _get_column_names(batch, postfix):
    return [column.name
            for column
            in _get_columns(batch, postfix)]


def _get_columns(batch, postfix):
    result = []
    for name in _AUGMENTABLE_NAMES:
        attr_name = name + postfix
        value = getattr(batch, name + postfix)
        # Every data item is either an array or a list. If there are no
        # items in the array/list, there are also no shapes to change
        # as shape-changes are imagewise. Hence, we can afford to check
        # len() here.
        if value is not None and len(value) > 0:
            result.append(_AugmentableColumn(name, value, attr_name))
    return result


# TODO also support (H,W,C) for heatmaps of len(images) == 1
# TODO also support (H,W) for segmaps of len(images) == 1
class UnnormalizedBatch(object):
    """
    Class for batches of unnormalized data before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or iterable of (H,W,C) ndarray or iterable of (H,W) ndarray
        The images to augment.

    heatmaps : None or (N,H,W,C) ndarray or imgaug.augmentables.heatmaps.HeatmapsOnImage or iterable of (H,W,C) ndarray or iterable of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.
        If anything else than ``HeatmapsOnImage``, then the number of heatmaps
        must match the number of images provided via parameter `images`.
        The number is contained either in ``N`` or the first iterable's size.

    segmentation_maps : None or (N,H,W) ndarray or imgaug.augmentables.segmaps.SegmentationMapsOnImage or iterable of (H,W) ndarray or iterable of imgaug.augmentables.segmaps.SegmentationMapsOnImage
        The segmentation maps to augment.
        If anything else than ``SegmentationMapsOnImage``, then the number of
        segmaps must match the number of images provided via parameter
        `images`. The number is contained either in ``N`` or the first
        iterable's size.

    keypoints : None or list of (N,K,2) ndarray or tuple of number or imgaug.augmentables.kps.Keypoint or iterable of (K,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.kps.KeypointOnImage or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint
        The keypoints to augment.
        If a tuple (or iterable(s) of tuple), then iterpreted as (x,y)
        coordinates and must hence contain two numbers.
        A single tuple represents a single coordinate on one image, an
        iterable of tuples the coordinates on one image and an iterable of
        iterable of tuples the coordinates on several images. Analogous if
        ``Keypoint`` objects are used instead of tuples.
        If an ndarray, then ``N`` denotes the number of images and ``K`` the
        number of keypoints on each image.
        If anything else than ``KeypointsOnImage`` is provided, then the
        number of keypoint groups must match the number of images provided
        via parameter `images`. The number is contained e.g. in ``N`` or
        in case of "iterable of iterable of tuples" in the first iterable's
        size.

    bounding_boxes : None or (N,B,4) ndarray or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of (B,4) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.bbs.BoundingBox or iterable of imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of iterable of tuple of number or iterable of iterable imgaug.augmentables.bbs.BoundingBox
        The bounding boxes to augment.
        This is analogous to the `keypoints` parameter. However, each
        tuple -- and also the last index in case of arrays -- has size 4,
        denoting the bounding box coordinates ``x1``, ``y1``, ``x2`` and ``y2``.

    polygons : None  or (N,#polys,#points,2) ndarray or imgaug.augmentables.polys.Polygon or imgaug.augmentables.polys.PolygonsOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.polys.Polygon or iterable of imgaug.augmentables.polys.PolygonsOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.polys.Polygon or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint
        The polygons to augment.
        This is similar to the `keypoints` parameter. However, each polygon
        may be made up of several ``(x,y)`` coordinates (three or more are
        required for valid polygons).
        The following datatypes will be interpreted as a single polygon on a
        single image:

          * ``imgaug.augmentables.polys.Polygon``
          * ``iterable of tuple of number``
          * ``iterable of imgaug.augmentables.kps.Keypoint``

        The following datatypes will be interpreted as multiple polygons on a
        single image:

          * ``imgaug.augmentables.polys.PolygonsOnImage``
          * ``iterable of imgaug.augmentables.polys.Polygon``
          * ``iterable of iterable of tuple of number``
          * ``iterable of iterable of imgaug.augmentables.kps.Keypoint``
          * ``iterable of iterable of imgaug.augmentables.polys.Polygon``

        The following datatypes will be interpreted as multiple polygons on
        multiple images:

          * ``(N,#polys,#points,2) ndarray``
          * ``iterable of (#polys,#points,2) ndarray``
          * ``iterable of iterable of (#points,2) ndarray``
          * ``iterable of iterable of iterable of tuple of number``
          * ``iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint``

    line_strings : None or (N,#lines,#points,2) ndarray or imgaug.augmentables.lines.LineString or imgaug.augmentables.lines.LineStringOnImage or iterable of (#lines,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.lines.LineString or iterable of imgaug.augmentables.lines.LineStringOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.polys.LineString or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint
        The line strings to augment.
        See `polygons` for more details as polygons follow a similar
        structure to line strings.

    data
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """

    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 line_strings=None, data=None):
        """Construct a new :class:`UnnormalizedBatch` instance."""
        self.images_unaug = images
        self.images_aug = None
        self.heatmaps_unaug = heatmaps
        self.heatmaps_aug = None
        self.segmentation_maps_unaug = segmentation_maps
        self.segmentation_maps_aug = None
        self.keypoints_unaug = keypoints
        self.keypoints_aug = None
        self.bounding_boxes_unaug = bounding_boxes
        self.bounding_boxes_aug = None
        self.polygons_unaug = polygons
        self.polygons_aug = None
        self.line_strings_unaug = line_strings
        self.line_strings_aug = None
        self.data = data

    def get_column_names(self):
        """Get the names of types of augmentables that contain data.

        This method is intended for situations where one wants to know which
        data is contained in the batch that has to be augmented, visualized
        or something similar.

        Added in 0.4.0.

        Returns
        -------
        list of str
            Names of types of augmentables. E.g. ``["images", "polygons"]``.

        """
        return _get_column_names(self, "_unaug")

    def to_normalized_batch(self):
        """Convert this unnormalized batch to an instance of Batch.

        As this method is intended to be called before augmentation, it
        assumes that none of the ``*_aug`` attributes is yet set.
        It will produce an AssertionError otherwise.

        The newly created Batch's ``*_unaug`` attributes will match the ones
        in this batch, just in normalized form.

        Returns
        -------
        imgaug.augmentables.batches.Batch
            The batch, with ``*_unaug`` attributes being normalized.

        """
        contains_no_augmented_data_yet = all([
            attr is None
            for attr_name, attr
            in self.__dict__.items()
            if attr_name.endswith("_aug")])
        assert contains_no_augmented_data_yet, (
            "Expected UnnormalizedBatch to not contain any augmented data "
            "before normalization, but at least one '*_aug' attribute was "
            "already set.")

        images_unaug = nlib.normalize_images(self.images_unaug)
        shapes = None
        if images_unaug is not None:
            shapes = [image.shape for image in images_unaug]

        return Batch(
            images=images_unaug,
            heatmaps=nlib.normalize_heatmaps(
                self.heatmaps_unaug, shapes),
            segmentation_maps=nlib.normalize_segmentation_maps(
                self.segmentation_maps_unaug, shapes),
            keypoints=nlib.normalize_keypoints(
                self.keypoints_unaug, shapes),
            bounding_boxes=nlib.normalize_bounding_boxes(
                self.bounding_boxes_unaug, shapes),
            polygons=nlib.normalize_polygons(
                self.polygons_unaug, shapes),
            line_strings=nlib.normalize_line_strings(
                self.line_strings_unaug, shapes),
            data=self.data
        )

    def fill_from_augmented_normalized_batch_(self, batch_aug_norm):
        """
        Fill this batch with (normalized) augmentation results in-place.

        This method receives a (normalized) Batch instance, takes all
        ``*_aug`` attributes out if it and assigns them to this
        batch *in unnormalized form*. Hence, the datatypes of all ``*_aug``
        attributes will match the datatypes of the ``*_unaug`` attributes.

        Added in 0.4.0.

        Parameters
        ----------
        batch_aug_norm: imgaug.augmentables.batches.Batch
            Batch after normalization and augmentation.

        Returns
        -------
        imgaug.augmentables.batches.UnnormalizedBatch
            This instance itself.
            All ``*_unaug`` attributes are unchanged.
            All ``*_aug`` attributes are taken from `batch_normalized`,
            converted to unnormalized form.

        """
        self.images_aug = nlib.invert_normalize_images(
            batch_aug_norm.images_aug, self.images_unaug)
        self.heatmaps_aug = nlib.invert_normalize_heatmaps(
            batch_aug_norm.heatmaps_aug, self.heatmaps_unaug)
        self.segmentation_maps_aug = nlib.invert_normalize_segmentation_maps(
            batch_aug_norm.segmentation_maps_aug, self.segmentation_maps_unaug)
        self.keypoints_aug = nlib.invert_normalize_keypoints(
            batch_aug_norm.keypoints_aug, self.keypoints_unaug)
        self.bounding_boxes_aug = nlib.invert_normalize_bounding_boxes(
            batch_aug_norm.bounding_boxes_aug, self.bounding_boxes_unaug)
        self.polygons_aug = nlib.invert_normalize_polygons(
            batch_aug_norm.polygons_aug, self.polygons_unaug)
        self.line_strings_aug = nlib.invert_normalize_line_strings(
            batch_aug_norm.line_strings_aug, self.line_strings_unaug)
        return self

    def fill_from_augmented_normalized_batch(self, batch_aug_norm):
        """
        Fill this batch with (normalized) augmentation results.

        This method receives a (normalized) Batch instance, takes all
        ``*_aug`` attributes out if it and assigns them to this
        batch *in unnormalized form*. Hence, the datatypes of all ``*_aug``
        attributes will match the datatypes of the ``*_unaug`` attributes.

        Parameters
        ----------
        batch_aug_norm: imgaug.augmentables.batches.Batch
            Batch after normalization and augmentation.

        Returns
        -------
        imgaug.augmentables.batches.UnnormalizedBatch
            New UnnormalizedBatch instance. All ``*_unaug`` attributes are
            taken from the old UnnormalizedBatch (without deepcopying them)
            and all ``*_aug`` attributes are taken from `batch_normalized`,
            converted to unnormalized form.

        """
        # we take here the .data from the normalized batch instead of from
        # self for the rare case where one has decided to somehow change it
        # during augmentation
        batch = UnnormalizedBatch(
            images=self.images_unaug,
            heatmaps=self.heatmaps_unaug,
            segmentation_maps=self.segmentation_maps_unaug,
            keypoints=self.keypoints_unaug,
            bounding_boxes=self.bounding_boxes_unaug,
            polygons=self.polygons_unaug,
            line_strings=self.line_strings_unaug,
            data=batch_aug_norm.data
        )

        batch.images_aug = nlib.invert_normalize_images(
            batch_aug_norm.images_aug, self.images_unaug)
        batch.heatmaps_aug = nlib.invert_normalize_heatmaps(
            batch_aug_norm.heatmaps_aug, self.heatmaps_unaug)
        batch.segmentation_maps_aug = nlib.invert_normalize_segmentation_maps(
            batch_aug_norm.segmentation_maps_aug, self.segmentation_maps_unaug)
        batch.keypoints_aug = nlib.invert_normalize_keypoints(
            batch_aug_norm.keypoints_aug, self.keypoints_unaug)
        batch.bounding_boxes_aug = nlib.invert_normalize_bounding_boxes(
            batch_aug_norm.bounding_boxes_aug, self.bounding_boxes_unaug)
        batch.polygons_aug = nlib.invert_normalize_polygons(
            batch_aug_norm.polygons_aug, self.polygons_unaug)
        batch.line_strings_aug = nlib.invert_normalize_line_strings(
            batch_aug_norm.line_strings_aug, self.line_strings_unaug)

        return batch


class Batch(object):
    """
    Class encapsulating a batch before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or list of (H,W,C) ndarray
        The images to augment.

    heatmaps : None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.

    segmentation_maps : None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
        The segmentation maps to augment.

    keypoints : None or list of imgaug.augmentables.kps.KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
        The bounding boxes to augment.

    polygons : None or list of imgaug.augmentables.polys.PolygonsOnImage
        The polygons to augment.

    line_strings : None or list of imgaug.augmentables.lines.LineStringsOnImage
        The line strings to augment.

    data
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.

    """

    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 line_strings=None, data=None):
        """Construct a new :class:`Batch` instance."""
        self.images_unaug = images
        self.images_aug = None
        self.heatmaps_unaug = heatmaps
        self.heatmaps_aug = None
        self.segmentation_maps_unaug = segmentation_maps
        self.segmentation_maps_aug = None
        self.keypoints_unaug = keypoints
        self.keypoints_aug = None
        self.bounding_boxes_unaug = bounding_boxes
        self.bounding_boxes_aug = None
        self.polygons_unaug = polygons
        self.polygons_aug = None
        self.line_strings_unaug = line_strings
        self.line_strings_aug = None
        self.data = data

    @property
    @ia.deprecated("Batch.images_unaug")
    def images(self):
        """Get unaugmented images."""
        return self.images_unaug

    @property
    @ia.deprecated("Batch.heatmaps_unaug")
    def heatmaps(self):
        """Get unaugmented heatmaps."""
        return self.heatmaps_unaug

    @property
    @ia.deprecated("Batch.segmentation_maps_unaug")
    def segmentation_maps(self):
        """Get unaugmented segmentation maps."""
        return self.segmentation_maps_unaug

    @property
    @ia.deprecated("Batch.keypoints_unaug")
    def keypoints(self):
        """Get unaugmented keypoints."""
        return self.keypoints_unaug

    @property
    @ia.deprecated("Batch.bounding_boxes_unaug")
    def bounding_boxes(self):
        """Get unaugmented bounding boxes."""
        return self.bounding_boxes_unaug

    def get_column_names(self):
        """Get the names of types of augmentables that contain data.

        This method is intended for situations where one wants to know which
        data is contained in the batch that has to be augmented, visualized
        or something similar.

        Added in 0.4.0.

        Returns
        -------
        list of str
            Names of types of augmentables. E.g. ``["images", "polygons"]``.

        """
        return _get_column_names(self, "_unaug")

    def to_normalized_batch(self):
        """Return this batch.

        This method does nothing and only exists to simplify interfaces
        that accept both :class:`UnnormalizedBatch` and :class:`Batch`.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.batches.Batch
            This batch (not copied).

        """
        return self

    def to_batch_in_augmentation(self):
        """Convert this batch to a :class:`_BatchInAugmentation` instance.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.batches._BatchInAugmentation
            The converted batch.

        """
        def _copy(var):
            # TODO first check here if _aug is set and if it is then use that?
            if var is not None:
                return utils.copy_augmentables(var)
            return var

        return _BatchInAugmentation(
            images=_copy(self.images_unaug),
            heatmaps=_copy(self.heatmaps_unaug),
            segmentation_maps=_copy(self.segmentation_maps_unaug),
            keypoints=_copy(self.keypoints_unaug),
            bounding_boxes=_copy(self.bounding_boxes_unaug),
            polygons=_copy(self.polygons_unaug),
            line_strings=_copy(self.line_strings_unaug)
        )

    def fill_from_batch_in_augmentation_(self, batch_in_augmentation):
        """Set the columns in this batch to the column values of another batch.

        This method works in-place.

        Added in 0.4.0.

        Parameters
        ----------
        batch_in_augmentation : _BatchInAugmentation
            Batch of which to use the column values.
            The values are *not* copied. Only their references are used.

        Returns
        -------
        Batch
            The updated batch. (Modified in-place.)

        """
        self.images_aug = batch_in_augmentation.images
        self.heatmaps_aug = batch_in_augmentation.heatmaps
        self.segmentation_maps_aug = batch_in_augmentation.segmentation_maps
        self.keypoints_aug = batch_in_augmentation.keypoints
        self.bounding_boxes_aug = batch_in_augmentation.bounding_boxes
        self.polygons_aug = batch_in_augmentation.polygons
        self.line_strings_aug = batch_in_augmentation.line_strings
        return self

    def deepcopy(self,
                 images_unaug=DEFAULT,
                 images_aug=DEFAULT,
                 heatmaps_unaug=DEFAULT,
                 heatmaps_aug=DEFAULT,
                 segmentation_maps_unaug=DEFAULT,
                 segmentation_maps_aug=DEFAULT,
                 keypoints_unaug=DEFAULT,
                 keypoints_aug=DEFAULT,
                 bounding_boxes_unaug=DEFAULT,
                 bounding_boxes_aug=DEFAULT,
                 polygons_unaug=DEFAULT,
                 polygons_aug=DEFAULT,
                 line_strings_unaug=DEFAULT,
                 line_strings_aug=DEFAULT):
        """Copy this batch and all of its column values.

        Parameters
        ----------
        images_unaug : imgaug.augmentables.batches.DEFAULT or None or (N,H,W,C) ndarray or list of (H,W,C) ndarray
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        images_aug : imgaug.augmentables.batches.DEFAULT or None or (N,H,W,C) ndarray or list of (H,W,C) ndarray
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        heatmaps_unaug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        heatmaps_aug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        segmentation_maps_unaug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        segmentation_maps_aug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        keypoints_unaug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.kps.KeypointOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        keypoints_aug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.kps.KeypointOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        bounding_boxes_unaug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        bounding_boxes_aug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        polygons_unaug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.polys.PolygonsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        polygons_aug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.polys.PolygonsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        line_strings_unaug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.lines.LineStringsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        line_strings_aug : imgaug.augmentables.batches.DEFAULT or None or list of imgaug.augmentables.lines.LineStringsOnImage
            Copies the current attribute value without changes if set to
            ``imgaug.augmentables.batches.DEFAULT``.
            Otherwise same as in :func:`Batch.__init__`.

        Returns
        -------
        Batch
            Deep copy of the batch, optionally with new attributes.

        """
        def _copy_optional(self_attr, arg):
            return utils.deepcopy_fast(arg if arg is not DEFAULT else self_attr)

        batch = Batch(
            images=_copy_optional(self.images_unaug, images_unaug),
            heatmaps=_copy_optional(self.heatmaps_unaug, heatmaps_unaug),
            segmentation_maps=_copy_optional(self.segmentation_maps_unaug,
                                             segmentation_maps_unaug),
            keypoints=_copy_optional(self.keypoints_unaug, keypoints_unaug),
            bounding_boxes=_copy_optional(self.bounding_boxes_unaug,
                                          bounding_boxes_unaug),
            polygons=_copy_optional(self.polygons_unaug, polygons_unaug),
            line_strings=_copy_optional(self.line_strings_unaug,
                                        line_strings_unaug),
            data=utils.deepcopy_fast(self.data)
        )
        batch.images_aug = _copy_optional(self.images_aug, images_aug)
        batch.heatmaps_aug = _copy_optional(self.heatmaps_aug, heatmaps_aug)
        batch.segmentation_maps_aug = _copy_optional(self.segmentation_maps_aug,
                                                     segmentation_maps_aug)
        batch.keypoints_aug = _copy_optional(self.keypoints_aug, keypoints_aug)
        batch.bounding_boxes_aug = _copy_optional(self.bounding_boxes_aug,
                                                  bounding_boxes_aug)
        batch.polygons_aug = _copy_optional(self.polygons_aug, polygons_aug)
        batch.line_strings_aug = _copy_optional(self.line_strings_aug,
                                                line_strings_aug)

        return batch


# Added in 0.4.0.
class _BatchInAugmentationPropagationContext(object):
    def __init__(self, batch, augmenter, hooks, parents):
        self.batch = batch
        self.augmenter = augmenter
        self.hooks = hooks
        self.parents = parents
        self.noned_info = None

    def __enter__(self):
        if self.hooks is not None:
            self.noned_info = self.batch.apply_propagation_hooks_(
                self.augmenter, self.hooks, self.parents)
        return self.batch

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.noned_info is not None:
            self.batch = \
                self.batch.invert_apply_propagation_hooks_(self.noned_info)


class _BatchInAugmentation(object):
    """
    Class encapsulating a batch during the augmentation process.

    Data within the batch is already verified and normalized, similar to
    :class:`Batch`. Data within the batch may be changed in-place. No initial
    copy is needed.

    Added in 0.4.0.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or list of (H,W,C) ndarray
        The images to augment.

    heatmaps : None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
        The heatmaps to augment.

    segmentation_maps : None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
        The segmentation maps to augment.

    keypoints : None or list of imgaug.augmentables.kps.KeypointOnImage
        The keypoints to augment.

    bounding_boxes : None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
        The bounding boxes to augment.

    polygons : None or list of imgaug.augmentables.polys.PolygonsOnImage
        The polygons to augment.

    line_strings : None or list of imgaug.augmentables.lines.LineStringsOnImage
        The line strings to augment.

    """

    # Added in 0.4.0.
    def __init__(self, images=None, heatmaps=None, segmentation_maps=None,
                 keypoints=None, bounding_boxes=None, polygons=None,
                 line_strings=None, data=None):
        """Create a new :class:`_BatchInAugmentation` instance."""
        self.images = images
        self.heatmaps = heatmaps
        self.segmentation_maps = segmentation_maps
        self.keypoints = keypoints
        self.bounding_boxes = bounding_boxes
        self.polygons = polygons
        self.line_strings = line_strings
        self.data = data

    @property
    def empty(self):
        """Estimate whether this batch is empty, i.e. contains no data.

        Added in 0.4.0.

        Returns
        -------
        bool
            ``True`` if the batch contains no data to augment.
            ``False`` otherwise.

        """
        return self.nb_rows == 0

    @property
    def nb_rows(self):
        """Get the number of rows (i.e. examples) in this batch.

        Note that this method assumes that all columns have the same number
        of rows.

        Added in 0.4.0.

        Returns
        -------
        int
            Number of rows or ``0`` if there is no data in the batch.

        """
        for augm_name in _AUGMENTABLE_NAMES:
            value = getattr(self, augm_name)
            if value is not None:
                return len(value)
        return 0

    @property
    def columns(self):
        """Get the columns of data to augment.

        Each column represents one datatype and its corresponding data,
        e.g. images or polygons.

        Added in 0.4.0.

        Returns
        -------
        list of _AugmentableColumn
            The columns to augment within this batch.

        """
        return _get_columns(self, "")

    def get_column_names(self):
        """Get the names of types of augmentables that contain data.

        This method is intended for situations where one wants to know which
        data is contained in the batch that has to be augmented, visualized
        or something similar.

        Added in 0.4.0.

        Returns
        -------
        list of str
            Names of types of augmentables. E.g. ``["images", "polygons"]``.

        """
        return _get_column_names(self, "")

    def get_rowwise_shapes(self):
        """Get the shape of each row within this batch.

        Each row denotes the data of different types (e.g. image array,
        polygons) corresponding to a single example in the batch.

        This method assumes that all ``.shape`` attributes contain the same
        shape and that it is identical to the image's shape.
        It also assumes that there are no columns containing only ``None`` s.

        Added in 0.4.0.

        Returns
        -------
        list of tuple of int
            The shapes of each row.

        """
        nb_rows = self.nb_rows
        columns = self.columns
        shapes = [None] * nb_rows
        found = np.zeros((nb_rows,), dtype=bool)
        for column in columns:
            if column.name == "images" and ia.is_np_array(column.value):
                shapes = [column.value.shape[1:]] * nb_rows
            else:
                for i, item in enumerate(column.value):
                    if item is not None:
                        shapes[i] = item.shape
                        found[i] = True
            if np.all(found):
                return shapes
        return shapes

    def subselect_rows_by_indices(self, indices):
        """Reduce this batch to a subset of rows based on their row indices.

        Added in 0.4.0.

        Parameters
        ----------
        indices : iterable of int
            Row indices to select.

        Returns
        -------
        _BatchInAugmentation
            Batch containing only a subselection of rows.

        """
        kwargs = {"data": self.data}
        for augm_name in _AUGMENTABLE_NAMES:
            rows = getattr(self, augm_name)
            if rows is not None:
                if augm_name == "images" and ia.is_np_array(rows):
                    rows = rows[indices]  # pylint: disable=unsubscriptable-object
                else:
                    rows = [rows[index] for index in indices]

                if len(rows) == 0:
                    rows = None
            kwargs[augm_name] = rows

        return _BatchInAugmentation(**kwargs)

    def invert_subselect_rows_by_indices_(self, indices, batch_subselected):
        """Reverse the subselection of rows in-place.

        This is the inverse of
        :func:`_BatchInAugmentation.subselect_rows_by_indices`.

        This method has to be executed on the batch *before* subselection.

        Added in 0.4.0.

        Parameters
        ----------
        indices : iterable of int
            Row indices that were selected. (This is the input to

        batch_subselected : _BatchInAugmentation
            The batch after
            :func:`_BatchInAugmentation.subselect_rows_by_indices` was called.

        Returns
        -------
        _BatchInAugmentation
            The updated batch. (Modified in-place.)

        Examples
        --------
        >>> import numpy as np
        >>> from imgaug.augmentables.batches import _BatchInAugmentation
        >>> images = np.zeros((2, 10, 20, 3), dtype=np.uint8)
        >>> batch = _BatchInAugmentation(images=images)
        >>> batch_sub = batch.subselect_rows_by_indices([0])
        >>> batch_sub.images += 1
        >>> batch = batch.invert_subselect_rows_by_indices_([0], batch_sub)

        """
        for augm_name in _AUGMENTABLE_NAMES:
            column = getattr(self, augm_name)
            if column is not None:
                column_sub = getattr(batch_subselected, augm_name)
                if column_sub is None:
                    # list of indices was empty, resulting in the columns
                    # in the subselected batch being empty and replaced
                    # by Nones. We can just re-use the columns before
                    # subselection.
                    pass
                elif augm_name == "images" and ia.is_np_array(column):
                    # An array does not have to stay an array after
                    # augmentation. The shapes and/or dtypes of rows may
                    # change, turning the array into a list.
                    if ia.is_np_array(column_sub):
                        shapes = {column.shape[1:], column_sub.shape[1:]}
                        dtypes = {column.dtype.name, column_sub.dtype.name}
                    else:
                        shapes = set(
                            [column.shape[1:]]
                            + [image.shape for image in column_sub])
                        dtypes = set(
                            [column.dtype.name]
                            + [image.dtype.name for image in column_sub])

                    if len(shapes) == 1 and len(dtypes) == 1:
                        column[indices] = column_sub  # pylint: disable=unsupported-assignment-operation
                    else:
                        self.images = list(column)
                        for ith_index, index in enumerate(indices):
                            self.images[index] = column_sub[ith_index]
                else:
                    for ith_index, index in enumerate(indices):
                        column[index] = column_sub[ith_index]  # pylint: disable=unsupported-assignment-operation

        return self

    def propagation_hooks_ctx(self, augmenter, hooks, parents):
        """Start a context in which propagation hooks are applied.

        Added in 0.4.0.

        Parameters
        ----------
        augmenter : imgaug.augmenters.meta.Augmenter
            Augmenter to provide to the propagation hook function.

        hooks : imgaug.imgaug.HooksImages or imgaug.imgaug.HooksKeypoints
            The hooks that might contain a propagation hook function.

        parents : list of imgaug.augmenters.meta.Augmenter
            The list of parents to provide to the propagation hook function.

        Returns
        -------
        _BatchInAugmentationPropagationContext
            The progagation hook context.

        """
        return _BatchInAugmentationPropagationContext(
            self, augmenter=augmenter, hooks=hooks, parents=parents)

    def apply_propagation_hooks_(self, augmenter, hooks, parents):
        """Set columns in this batch to ``None`` based on a propagation hook.

        This method works in-place.

        Added in 0.4.0.

        Parameters
        ----------
        augmenter : imgaug.augmenters.meta.Augmenter
            Augmenter to provide to the propagation hook function.

        hooks : imgaug.imgaug.HooksImages or imgaug.imgaug.HooksKeypoints
            The hooks that might contain a propagation hook function.

        parents : list of imgaug.augmenters.meta.Augmenter
            The list of parents to provide to the propagation hook function.

        Returns
        -------
        list of tuple of str
            Information about which columns were set to ``None``.
            Each tuple contains
            ``(column attribute name, column value before setting it to None)``.
            This information is required when calling
            :func:`_BatchInAugmentation.invert_apply_propagation_hooks_`.

        """
        if hooks is None:
            return None

        noned_info = []
        for column in self.columns:
            is_prop = hooks.is_propagating(
                column.value, augmenter=augmenter, parents=parents,
                default=True)
            if not is_prop:
                setattr(self, column.attr_name, None)
                noned_info.append((column.attr_name, column.value))
        return noned_info

    def invert_apply_propagation_hooks_(self, noned_info):
        """Set columns from ``None`` back to their original values.

        This is the inverse of
        :func:`_BatchInAugmentation.apply_propagation_hooks_`.

        This method works in-place.

        Added in 0.4.0.

        Parameters
        ----------
        noned_info : list of tuple of str
            Information about which columns were set to ``None`` and their
            original values. This is the output of
            :func:`_BatchInAugmentation.apply_propagation_hooks_`.

        Returns
        -------
        _BatchInAugmentation
            The updated batch. (Modified in-place.)

        """
        for attr_name, value in noned_info:
            setattr(self, attr_name, value)
        return self

    def to_batch_in_augmentation(self):
        """Convert this batch to a :class:`_BatchInAugmentation` instance.

        This method simply returns the batch itself. It exists for consistency
        with the other batch classes.

        Added in 0.4.0.

        Returns
        -------
        imgaug.augmentables.batches._BatchInAugmentation
            The batch itself. (Not copied.)

        """
        return self

    def fill_from_batch_in_augmentation_(self, batch_in_augmentation):
        """Set the columns in this batch to the column values of another batch.

        This method works in-place.

        Added in 0.4.0.

        Parameters
        ----------
        batch_in_augmentation : _BatchInAugmentation
            Batch of which to use the column values.
            The values are *not* copied. Only their references are used.

        Returns
        -------
        _BatchInAugmentation
            The updated batch. (Modified in-place.)

        """
        if batch_in_augmentation is self:
            return self

        self.images = batch_in_augmentation.images
        self.heatmaps = batch_in_augmentation.heatmaps
        self.segmentation_maps = batch_in_augmentation.segmentation_maps
        self.keypoints = batch_in_augmentation.keypoints
        self.bounding_boxes = batch_in_augmentation.bounding_boxes
        self.polygons = batch_in_augmentation.polygons
        self.line_strings = batch_in_augmentation.line_strings

        return self

    def to_batch(self, batch_before_aug):
        """Convert this batch into a :class:`Batch` instance.

        Added in 0.4.0.

        Parameters
        ----------
        batch_before_aug : imgaug.augmentables.batches.Batch
            The batch before augmentation. It is required to set the input
            data of the :class:`Batch` instance, e.g. ``images_unaug``
            or ``data``.

        Returns
        -------
        imgaug.augmentables.batches.Batch
            Batch, with original unaugmented inputs from `batch_before_aug`
            and augmented outputs from this :class:`_BatchInAugmentation`
            instance.

        """
        batch = Batch(
            images=batch_before_aug.images_unaug,
            heatmaps=batch_before_aug.heatmaps_unaug,
            segmentation_maps=batch_before_aug.segmentation_maps_unaug,
            keypoints=batch_before_aug.keypoints_unaug,
            bounding_boxes=batch_before_aug.bounding_boxes_unaug,
            polygons=batch_before_aug.polygons_unaug,
            line_strings=batch_before_aug.line_strings_unaug,
            data=batch_before_aug.data
        )
        batch.images_aug = self.images
        batch.heatmaps_aug = self.heatmaps
        batch.segmentation_maps_aug = self.segmentation_maps
        batch.keypoints_aug = self.keypoints
        batch.bounding_boxes_aug = self.bounding_boxes
        batch.polygons_aug = self.polygons
        batch.line_strings_aug = self.line_strings
        return batch

    def deepcopy(self):
        """Copy this batch and all of its column values.

        Added in 0.4.0.

        Returns
        -------
        _BatchInAugmentation
            Deep copy of this batch.

        """
        batch = _BatchInAugmentation(data=utils.deepcopy_fast(self.data))

        for augm_name in _AUGMENTABLE_NAMES:
            value = getattr(self, augm_name)
            if value is not None:
                setattr(batch, augm_name, utils.copy_augmentables(value))

        return batch
