"""
Augmenters that don't apply augmentations themselves, but are needed
for meta usage.

List of augmenters:

    * :class:`Augmenter` (base class for all augmenters)
    * :class:`Sequential`
    * :class:`SomeOf`
    * :class:`OneOf`
    * :class:`Sometimes`
    * :class:`WithChannels`
    * :class:`Identity`
    * :class:`Noop`
    * :class:`Lambda`
    * :class:`AssertLambda`
    * :class:`AssertShape`
    * :class:`ChannelShuffle`

Note: :class:`~imgaug.augmenters.color.WithColorspace` is in ``color.py``.

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod
import copy as copy_module
import re
import itertools
import functools
import sys

import numpy as np
import six
import six.moves as sm

import imgaug as ia
from imgaug.augmentables.batches import (Batch, UnnormalizedBatch,
                                         _BatchInAugmentation)
from .. import parameters as iap
from .. import random as iarandom
from . import base as iabase


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_image_(image, min_value, max_value):
    """Clip image in-place."""
    return clip_augmented_images_(image, min_value, max_value)


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_image(image, min_value, max_value):
    """Clip image."""
    return clip_augmented_images(image, min_value, max_value)


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_images_(images, min_value, max_value):
    """Clip images in-place."""
    if ia.is_np_array(images):
        return np.clip(images, min_value, max_value, out=images)
    return [np.clip(image, min_value, max_value, out=image)
            for image in images]


@ia.deprecated("imgaug.dtypes.clip_")
def clip_augmented_images(images, min_value, max_value):
    """Clip images."""
    if ia.is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return clip_augmented_images_(images, min_value, max_value)


def handle_children_list(lst, augmenter_name, lst_name, default="sequential"):
    """Normalize an augmenter list provided by a user."""
    if lst is None:
        if default == "sequential":
            return Sequential([], name="%s-%s" % (augmenter_name, lst_name))
        return default
    if isinstance(lst, Augmenter):
        if ia.is_iterable(lst):
            # TODO why was this assert added here? seems to make no sense
            only_augmenters = all([isinstance(child, Augmenter)
                                   for child in lst])
            assert only_augmenters, (
                "Expected all children to be augmenters, got types %s." % (
                    ", ".join([str(type(v)) for v in lst])))
            return lst
        return Sequential(lst, name="%s-%s" % (augmenter_name, lst_name))
    if ia.is_iterable(lst):
        if len(lst) == 0 and default != "sequential":
            return default
        only_augmenters = all([isinstance(child, Augmenter)
                               for child in lst])
        assert only_augmenters, (
            "Expected all children to be augmenters, got types %s." % (
                ", ".join([str(type(v)) for v in lst])))
        return Sequential(lst, name="%s-%s" % (augmenter_name, lst_name))
    raise Exception(
        "Expected None, Augmenter or list/tuple as children list %s "
        "for augmenter with name %s, got %s." % (
            lst_name, augmenter_name, type(lst),))


def reduce_to_nonempty(objs):
    """Remove from a list all objects that don't follow ``obj.empty==True``."""
    objs_reduced = []
    ids = []
    for i, obj in enumerate(objs):
        assert hasattr(obj, "empty"), (
            "Expected object with property 'empty'. Got type %s." % (
                type(obj),))
        if not obj.empty:
            objs_reduced.append(obj)
            ids.append(i)
    return objs_reduced, ids


def invert_reduce_to_nonempty(objs, ids, objs_reduced):
    """Inverse of :func:`reduce_to_nonempty`."""
    objs_inv = list(objs)
    for idx, obj_from_reduced in zip(ids, objs_reduced):
        objs_inv[idx] = obj_from_reduced
    return objs_inv


def estimate_max_number_of_channels(images):
    """Compute the maximum number of image channels among a list of images."""
    if ia.is_np_array(images):
        assert images.ndim == 4, (
            "Expected 'images' to be 4-dimensional if provided as array. "
            "Got %d dimensions." % (images.ndim,))
        return images.shape[3]

    assert ia.is_iterable(images), (
        "Expected 'images' to be an array or iterable, got %s." % (
            type(images),))
    if len(images) == 0:
        return None
    channels = [el.shape[2] if len(el.shape) >= 3 else 1 for el in images]
    return max(channels)


def copy_arrays(arrays):
    """Copy the arrays of a single input array or list of input arrays."""
    if ia.is_np_array(arrays):
        return np.copy(arrays)
    return [np.copy(array) for array in arrays]


def _add_channel_axis(arrs):
    if ia.is_np_array(arrs):
        if arrs.ndim == 3:  # (N,H,W)
            return arrs[..., np.newaxis]  # (N,H,W) -> (N,H,W,1)
        return arrs
    return [
        arr[..., np.newaxis]  # (H,W) -> (H,W,1)
        if arr.ndim == 2
        else arr
        for arr in arrs
    ]


def _remove_added_channel_axis(arrs_added, arrs_orig):
    if ia.is_np_array(arrs_orig):
        if arrs_orig.ndim == 3:  # (N,H,W)
            if ia.is_np_array(arrs_added):
                return arrs_added[..., 0]  # (N,H,W,1) -> (N,H,W)
            # (N,H,W) -> (N,H,W,1) -> <augmentation> -> list of (H,W,1)
            return [arr[..., 0] for arr in arrs_added]
        return arrs_added
    return [
        arr_added[..., 0]
        if arr_orig.ndim == 2
        else arr_added   # (H,W,1) -> (H,W)
        for arr_added, arr_orig
        in zip(arrs_added, arrs_orig)
    ]


class _maybe_deterministic_ctx(object):  # pylint: disable=invalid-name
    """Context that resets an RNG to its initial state upon exit.

    This allows to execute some sampling functions and leave the code block
    with the used RNG in the same state as before.

    Parameters
    ----------
    random_state : imgaug.random.RNG or imgaug.augmenters.meta.Augmenter
        The RNG to reset. If this is an augmenter, then the augmenter's
        RNG will be used.

    deterministic : None or bool
        Whether to reset the RNG upon exit (``True``) or not (``False``).
        Allowed to be ``None`` iff `random_state` was an augmenter, in which
        case that augmenter's ``deterministic`` attribute will be used.

    """

    def __init__(self, random_state, deterministic=None):
        if deterministic is None:
            augmenter = random_state
            self.random_state = augmenter.random_state
            self.deterministic = augmenter.deterministic
        else:
            assert deterministic is not None, (
                "Expected boolean as `deterministic`, got None.")
            self.random_state = random_state
            self.deterministic = deterministic
        self.old_state = None

    def __enter__(self):
        if self.deterministic:
            self.old_state = self.random_state.state

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.old_state is not None:
            self.random_state.state = self.old_state


@six.add_metaclass(ABCMeta)
class Augmenter(object):
    """
    Base class for Augmenter objects.
    All augmenters derive from this class.

    Parameters
    ----------
    seed : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Seed to use for this augmenter's random number generator (RNG) or
        alternatively an RNG itself. Setting this parameter allows to
        control/influence the random number sampling of this specific
        augmenter without affecting other augmenters. Usually, there is no
        need to set this parameter.

            * If ``None``: The global RNG is used (shared by all
              augmenters).
            * If ``int``: The value will be used as a seed for a new
              :class:`~imgaug.random.RNG` instance.
            * If :class:`~imgaug.random.RNG`: The ``RNG`` instance will be
              used without changes.
            * If :class:`~imgaug.random.Generator`: A new
              :class:`~imgaug.random.RNG` instance will be
              created, containing that generator.
            * If :class:`~imgaug.random.bit_generator.BitGenerator`: Will
              be wrapped in a :class:`~imgaug.random.Generator`. Then
              similar behaviour to :class:`~imgaug.random.Generator`
              parameters.
            * If :class:`~imgaug.random.SeedSequence`: Will
              be wrapped in a new bit generator and
              :class:`~imgaug.random.Generator`. Then
              similar behaviour to :class:`~imgaug.random.Generator`
              parameters.
            * If :class:`~imgaug.random.RandomState`: Similar behaviour to
              :class:`~imgaug.random.Generator`. Outdated in numpy 1.17+.

        If a new bit generator has to be created, it will be an instance
        of :class:`numpy.random.SFC64`.

        Added in 0.4.0.

    name : None or str, optional
        Name given to the Augmenter instance. This name is used when
        converting the instance to a string, e.g. for ``print`` statements.
        It is also used for ``find``, ``remove`` or similar operations
        on augmenters with children.
        If ``None``, ``UnnamedX`` will be used as the name, where ``X``
        is the Augmenter's class name.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    """

    def __init__(self, seed=None, name=None,
                 random_state="deprecated",
                 deterministic="deprecated"):
        """Create a new Augmenter instance."""
        super(Augmenter, self).__init__()

        assert name is None or ia.is_string(name), (
            "Expected name to be None or string-like, got %s." % (
                type(name),))
        if name is None:
            self.name = "Unnamed%s" % (self.__class__.__name__,)
        else:
            self.name = name

        if deterministic != "deprecated":
            ia.warn_deprecated(
                "The parameter `deterministic` is deprecated "
                "in `imgaug.augmenters.meta.Augmenter`. Use "
                "`.to_deterministic()` to switch into deterministic mode.",
                stacklevel=4)
            assert ia.is_single_bool(deterministic), (
                "Expected deterministic to be a boolean, got %s." % (
                    type(deterministic),))
        else:
            deterministic = False

        self.deterministic = deterministic

        if random_state != "deprecated":
            assert seed is None, "Cannot set both `seed` and `random_state`."
            seed = random_state

        if deterministic and seed is None:
            # Usually if None is provided, the global RNG will be used.
            # In case of deterministic mode we most likely rather want a local
            # RNG, which is here created.
            self.random_state = iarandom.RNG.create_pseudo_random_()
        else:
            # self.random_state = iarandom.normalize_rng_(random_state)
            self.random_state = iarandom.RNG(seed)

        self.activated = True

    def augment_batches(self, batches, hooks=None, background=False):
        """Augment multiple batches.

        In contrast to other ``augment_*`` method, this one **yields**
        batches instead of returning a full list. This is more suited
        for most training loops.

        This method also also supports augmentation on multiple cpu cores,
        activated via the `background` flag. If the `background` flag
        is activated, an instance of :class:`~imgaug.multicore.Pool` will
        be spawned using all available logical CPU cores and an
        ``output_buffer_size`` of ``C*10``, where ``C`` is the number of
        logical CPU cores. I.e. a maximum of ``C*10`` batches will be somewhere
        in the augmentation pipeline (or waiting to be retrieved by downstream
        functions) before this method temporarily stops the loading of new
        batches from `batches`.

        Parameters
        ----------
        batches : imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch or iterable of imgaug.augmentables.batches.Batch or iterable of imgaug.augmentables.batches.UnnormalizedBatch
            A single batch or a list of batches to augment.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        background : bool, optional
            Whether to augment the batches in background processes.
            If ``True``, hooks can currently not be used as that would require
            pickling functions.
            Note that multicore augmentation distributes the batches onto
            different CPU cores. It does *not* split the data *within* batches.
            It is therefore *not* sensible to use ``background=True`` to
            augment a single batch. Only use it for multiple batches.
            Note also that multicore augmentation needs some time to start. It
            is therefore not recommended to use it for very few batches.

        Yields
        -------
        imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch or iterable of imgaug.augmentables.batches.Batch or iterable of imgaug.augmentables.batches.UnnormalizedBatch
            Augmented batches.

        """
        if isinstance(batches, (Batch, UnnormalizedBatch)):
            batches = [batches]

        assert (
            (ia.is_iterable(batches)
             and not ia.is_np_array(batches)
             and not ia.is_string(batches))
            or ia.is_generator(batches)), (
                "Expected either (a) an iterable that is not an array or a "
                "string or (b) a generator. Got: %s" % (type(batches),))

        if background:
            assert hooks is None, (
                "Hooks can not be used when background augmentation is "
                "activated.")

        def _normalize_batch(idx, batch):
            if isinstance(batch, Batch):
                batch_copy = batch.deepcopy()
                batch_copy.data = (idx, batch_copy.data)
                batch_normalized = batch_copy
                batch_orig_dt = "imgaug.Batch"
            elif isinstance(batch, UnnormalizedBatch):
                batch_copy = batch.to_normalized_batch()
                batch_copy.data = (idx, batch_copy.data)
                batch_normalized = batch_copy
                batch_orig_dt = "imgaug.UnnormalizedBatch"
            elif ia.is_np_array(batch):
                assert batch.ndim in (3, 4), (
                    "Expected numpy array to have shape (N, H, W) or "
                    "(N, H, W, C), got %s." % (batch.shape,))
                batch_normalized = Batch(images=batch, data=(idx,))
                batch_orig_dt = "numpy_array"
            elif isinstance(batch, list):
                if len(batch) == 0:
                    batch_normalized = Batch(data=(idx,))
                    batch_orig_dt = "empty_list"
                elif ia.is_np_array(batch[0]):
                    batch_normalized = Batch(images=batch, data=(idx,))
                    batch_orig_dt = "list_of_numpy_arrays"
                elif isinstance(batch[0], ia.HeatmapsOnImage):
                    batch_normalized = Batch(heatmaps=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.HeatmapsOnImage"
                elif isinstance(batch[0], ia.SegmentationMapsOnImage):
                    batch_normalized = Batch(segmentation_maps=batch,
                                             data=(idx,))
                    batch_orig_dt = "list_of_imgaug.SegmentationMapsOnImage"
                elif isinstance(batch[0], ia.KeypointsOnImage):
                    batch_normalized = Batch(keypoints=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.KeypointsOnImage"
                elif isinstance(batch[0], ia.BoundingBoxesOnImage):
                    batch_normalized = Batch(bounding_boxes=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.BoundingBoxesOnImage"
                elif isinstance(batch[0], ia.PolygonsOnImage):
                    batch_normalized = Batch(polygons=batch, data=(idx,))
                    batch_orig_dt = "list_of_imgaug.PolygonsOnImage"
                else:
                    raise Exception(
                        "Unknown datatype in batch[0]. Expected numpy array "
                        "or imgaug.HeatmapsOnImage or "
                        "imgaug.SegmentationMapsOnImage or "
                        "imgaug.KeypointsOnImage or "
                        "imgaug.BoundingBoxesOnImage, "
                        "or imgaug.PolygonsOnImage, "
                        "got %s." % (type(batch[0]),))
            else:
                raise Exception(
                    "Unknown datatype of batch. Expected imgaug.Batch or "
                    "imgaug.UnnormalizedBatch or "
                    "numpy array or list of (numpy array or "
                    "imgaug.HeatmapsOnImage or "
                    "imgaug.SegmentationMapsOnImage "
                    "or imgaug.KeypointsOnImage or "
                    "imgaug.BoundingBoxesOnImage or "
                    "imgaug.PolygonsOnImage). Got %s." % (type(batch),))

            if batch_orig_dt not in ["imgaug.Batch",
                                     "imgaug.UnnormalizedBatch"]:
                ia.warn_deprecated(
                    "Received an input in augment_batches() that was not an "
                    "instance of imgaug.augmentables.batches.Batch "
                    "or imgaug.augmentables.batches.UnnormalizedBatch, but "
                    "instead %s. This is deprecated. Use augment() for such "
                    "data or wrap it in a Batch instance." % (
                        batch_orig_dt,))
            return batch_normalized, batch_orig_dt

        # unnormalization of non-Batch/UnnormalizedBatch is for legacy support
        def _unnormalize_batch(batch_aug, batch_orig, batch_orig_dt):
            if batch_orig_dt == "imgaug.Batch":
                batch_unnormalized = batch_aug
                # change (i, .data) back to just .data
                batch_unnormalized.data = batch_unnormalized.data[1]
            elif batch_orig_dt == "imgaug.UnnormalizedBatch":
                # change (i, .data) back to just .data
                batch_aug.data = batch_aug.data[1]

                batch_unnormalized = \
                    batch_orig.fill_from_augmented_normalized_batch(batch_aug)
            elif batch_orig_dt == "numpy_array":
                batch_unnormalized = batch_aug.images_aug
            elif batch_orig_dt == "empty_list":
                batch_unnormalized = []
            elif batch_orig_dt == "list_of_numpy_arrays":
                batch_unnormalized = batch_aug.images_aug
            elif batch_orig_dt == "list_of_imgaug.HeatmapsOnImage":
                batch_unnormalized = batch_aug.heatmaps_aug
            elif batch_orig_dt == "list_of_imgaug.SegmentationMapsOnImage":
                batch_unnormalized = batch_aug.segmentation_maps_aug
            elif batch_orig_dt == "list_of_imgaug.KeypointsOnImage":
                batch_unnormalized = batch_aug.keypoints_aug
            elif batch_orig_dt == "list_of_imgaug.BoundingBoxesOnImage":
                batch_unnormalized = batch_aug.bounding_boxes_aug
            else:  # only option left
                assert batch_orig_dt == "list_of_imgaug.PolygonsOnImage", (
                    "Got an unexpected type %s." % (type(batch_orig_dt),))
                batch_unnormalized = batch_aug.polygons_aug
            return batch_unnormalized

        if not background:
            # singlecore augmentation

            for idx, batch in enumerate(batches):
                batch_normalized, batch_orig_dt = _normalize_batch(idx, batch)
                batch_normalized = self.augment_batch_(
                    batch_normalized, hooks=hooks)
                batch_unnormalized = _unnormalize_batch(
                    batch_normalized, batch, batch_orig_dt)

                yield batch_unnormalized
        else:
            # multicore augmentation
            import imgaug.multicore as multicore

            id_to_batch_orig = dict()

            def load_batches():
                for idx, batch in enumerate(batches):
                    batch_normalized, batch_orig_dt = _normalize_batch(
                        idx, batch)
                    id_to_batch_orig[idx] = (batch, batch_orig_dt)
                    yield batch_normalized

            with multicore.Pool(self) as pool:
                # pylint:disable=protected-access
                # note that pool.processes is None here
                output_buffer_size = pool.pool._processes * 10

                for batch_aug in pool.imap_batches(
                        load_batches(), output_buffer_size=output_buffer_size):
                    idx = batch_aug.data[0]
                    assert idx in id_to_batch_orig, (
                        "Got idx %d from Pool, which is not known." % (
                            idx))
                    batch_orig, batch_orig_dt = id_to_batch_orig[idx]
                    batch_unnormalized = _unnormalize_batch(
                        batch_aug, batch_orig, batch_orig_dt)
                    del id_to_batch_orig[idx]
                    yield batch_unnormalized

    # we deprecate here so that users switch to `augment_batch_()` and in the
    # future we can add a `parents` parameter here without having to consider
    # that a breaking change
    @ia.deprecated("augment_batch_()",
                   comment="`augment_batch()` was renamed to "
                           "`augment_batch_()` as it changes all `*_unaug` "
                           "attributes of batches in-place. Note that "
                           "`augment_batch_()` has now a `parents` parameter. "
                           "Calls of the style `augment_batch(batch, hooks)` "
                           "must be changed to "
                           "`augment_batch(batch, hooks=hooks)`.")
    def augment_batch(self, batch, hooks=None):
        """Augment a single batch.

        Deprecated since 0.4.0.

        """
        # We call augment_batch_() directly here without copy, because this
        # method never copies. Would make sense to add a copy here if the
        # method is un-deprecated at some point.
        return self.augment_batch_(batch, hooks=hooks)

    # TODO add more tests
    def augment_batch_(self, batch, parents=None, hooks=None):
        """
        Augment a single batch in-place.

        Added in 0.4.0.

        Parameters
        ----------
        batch : imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch or imgaug.augmentables.batch._BatchInAugmentation
            A single batch to augment.

            If :class:`imgaug.augmentables.batches.UnnormalizedBatch`
            or :class:`imgaug.augmentables.batches.Batch`, then the ``*_aug``
            attributes may be modified in-place, while the ``*_unaug``
            attributes will not be modified.
            If :class:`imgaug.augmentables.batches._BatchInAugmentation`,
            then all attributes may be modified in-place.

        parents : None or list of imgaug.augmenters.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        imgaug.augmentables.batches.Batch or imgaug.augmentables.batches.UnnormalizedBatch
            Augmented batch.

        """
        # this chain of if/elses would be more beautiful if it was
        # (1st) UnnormalizedBatch, (2nd) Batch, (3rd) BatchInAugmenation.
        # We check for _BatchInAugmentation first as it is expected to be the
        # most common input (due to child calls).
        batch_unnorm = None
        batch_norm = None
        if isinstance(batch, _BatchInAugmentation):
            batch_inaug = batch
        elif isinstance(batch, UnnormalizedBatch):
            batch_unnorm = batch
            batch_norm = batch.to_normalized_batch()
            batch_inaug = batch_norm.to_batch_in_augmentation()
        elif isinstance(batch, Batch):
            batch_norm = batch
            batch_inaug = batch_norm.to_batch_in_augmentation()
        else:
            raise ValueError(
                "Expected UnnormalizedBatch, Batch or _BatchInAugmentation, "
                "got %s." % (type(batch).__name__,))

        columns = batch_inaug.columns

        # hooks preprocess
        if hooks is not None:
            for column in columns:
                value = hooks.preprocess(
                    column.value, augmenter=self, parents=parents)
                setattr(batch_inaug, column.attr_name, value)

            # refresh so that values are updated for later functions
            columns = batch_inaug.columns

        # set augmentables to None if this augmenter is deactivated or hooks
        # demands it
        set_to_none = []
        if not self.activated:
            for column in columns:
                set_to_none.append(column)
                setattr(batch_inaug, column.attr_name, None)
        elif hooks is not None:
            for column in columns:
                activated = hooks.is_activated(
                    column.value, augmenter=self, parents=parents,
                    default=self.activated)
                if not activated:
                    set_to_none.append(column)
                    setattr(batch_inaug, column.attr_name, None)

        # If _augment_batch_() follows legacy-style and ends up calling
        # _augment_images() and similar methods, we don't need the
        # deterministic context here. But if there is a custom implementation
        # of _augment_batch_(), then we should have this here. It causes very
        # little overhead.
        with _maybe_deterministic_ctx(self):
            if not batch_inaug.empty:
                batch_inaug = self._augment_batch_(
                    batch_inaug,
                    random_state=self.random_state,
                    parents=parents if parents is not None else [],
                    hooks=hooks)

        # revert augmentables being set to None for non-activated augmenters
        for column in set_to_none:
            setattr(batch_inaug, column.attr_name, column.value)

        # hooks postprocess
        if hooks is not None:
            # refresh as contents may have been changed in _augment_batch_()
            columns = batch_inaug.columns

            for column in columns:
                augm_value = hooks.postprocess(
                    column.value, augmenter=self, parents=parents)
                setattr(batch_inaug, column.attr_name, augm_value)

        if batch_unnorm is not None:
            batch_norm = batch_norm.fill_from_batch_in_augmentation_(
                batch_inaug)
            batch_unnorm = batch_unnorm.fill_from_augmented_normalized_batch_(
                batch_norm)
            return batch_unnorm
        if batch_norm is not None:
            batch_norm = batch_norm.fill_from_batch_in_augmentation_(
                batch_inaug)
            return batch_norm
        return batch_inaug

    def _augment_batch_(self, batch, random_state, parents, hooks):
        """Augment a single batch in-place.

        This is the internal version of :func:`Augmenter.augment_batch_`.
        It is called from :func:`Augmenter.augment_batch_` and should usually
        not be called directly.
        This method may transform the batches in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        Added in 0.4.0.

        Parameters
        ----------
        batch : imgaug.augmentables.batches._BatchInAugmentation
            The normalized batch to augment. May be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_batch_`.

        hooks : imgaug.imgaug.HooksImages or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_batch_`.

        Returns
        ----------
        imgaug.augmentables.batches._BatchInAugmentation
            The augmented batch.

        """
        # The code below covers the case of older augmenters that still have
        # _augment_images(), _augment_keypoints(), ... methods that augment
        # each input type on its own (including re-sampling from random
        # variables). The code block can be safely overwritten by a method
        # augmenting a whole batch of data in one step.

        columns = batch.columns
        multiple_columns = len(columns) > 1

        # For multi-column data (e.g. images + BBs) we need deterministic mode
        # within this batch, otherwise the datatypes within this batch would
        # get different samples.
        deterministic = self.deterministic or multiple_columns

        # set attribute batch.T_aug with result of self.augment_T() for each
        # batch.T_unaug (that had any content)
        for column in columns:
            with _maybe_deterministic_ctx(random_state, deterministic):
                value = getattr(self, "_augment_" + column.name)(
                    column.value, random_state=random_state,
                    parents=parents, hooks=hooks)
                setattr(batch, column.attr_name, value)

        # If the augmenter was alread in deterministic mode, we can expect
        # that to_deterministic() was called, which advances the RNG. But
        # if it wasn't and we had to auto-switch for the batch, there was not
        # advancement yet.
        if multiple_columns and not self.deterministic:
            random_state.advance_()

        return batch

    def augment_image(self, image, hooks=None):
        """Augment a single image.

        Parameters
        ----------
        image : (H,W,C) ndarray or (H,W) ndarray
            The image to augment.
            Channel-axis is optional, but expected to be the last axis if
            present. In most cases, this array should be of dtype ``uint8``,
            which is supported by all augmenters. Support for other dtypes
            varies by augmenter -- see the respective augmenter-specific
            documentation for more details.

        hooks : None or imgaug.HooksImages, optional
            HooksImages object to dynamically interfere with the augmentation
            process.

        Returns
        -------
        ndarray
            The corresponding augmented image.

        """
        assert ia.is_np_array(image), (
            "Expected to get a single numpy array of shape (H,W) or (H,W,C) "
            "for `image`. Got instead type %d. Use `augment_images(images)` "
            "to augment a list of multiple images." % (
                type(image).__name__),)
        assert image.ndim in [2, 3], (
            "Expected image to have shape (height, width, [channels]), "
            "got shape %s." % (image.shape,))
        iabase._warn_on_suspicious_single_image_shape(image)
        return self.augment_images([image], hooks=hooks)[0]

    def augment_images(self, images, parents=None, hooks=None):
        """Augment a batch of images.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            Images to augment.
            The input can be a list of numpy arrays or a single array. Each
            array is expected to have shape ``(H, W, C)`` or ``(H, W)``,
            where ``H`` is the height, ``W`` is the width and ``C`` are the
            channels. The number of channels may differ between images.
            If a list is provided, the height, width and channels may differ
            between images within the provided batch.
            In most cases, the image array(s) should be of dtype ``uint8``,
            which is supported by all augmenters. Support for other dtypes
            varies by augmenter -- see the respective augmenter-specific
            documentation for more details.

        parents : None or list of imgaug.augmenters.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksImages, optional
            :class:`~imgaug.imgaug.HooksImages` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        ndarray or list
            Corresponding augmented images.
            If the input was an ``ndarray``, the output is also an ``ndarray``,
            unless the used augmentations have led to different output image
            sizes (as can happen in e.g. cropping).

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> import numpy as np
        >>> aug = iaa.GaussianBlur((0.0, 3.0))
        >>> # create empty example images
        >>> images = np.zeros((2, 64, 64, 3), dtype=np.uint8)
        >>> images_aug = aug.augment_images(images)

        Create ``2`` empty (i.e. black) example numpy images and apply
        gaussian blurring to them.

        """
        iabase._warn_on_suspicious_multi_image_shapes(images)
        return self.augment_batch_(
            UnnormalizedBatch(images=images),
            parents=parents,
            hooks=hooks
        ).images_aug

    def _augment_images(self, images, random_state, parents, hooks):
        """Augment a batch of images in-place.

        This is the internal version of :func:`Augmenter.augment_images`.
        It is called from :func:`Augmenter.augment_images` and should usually
        not be called directly.
        It has to be implemented by every augmenter.
        This method may transform the images in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        images : (N,H,W,C) ndarray or list of (H,W,C) ndarray
            Images to augment.
            They may be changed in-place.
            Either a list of ``(H, W, C)`` arrays or a single ``(N, H, W, C)``
            array, where ``N`` is the number of images, ``H`` is the height of
            images, ``W`` is the width of images and ``C`` is the number of
            channels of images. In the case of a list as input, ``H``, ``W``
            and ``C`` may change per image.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_images`.

        hooks : imgaug.imgaug.HooksImages or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_images`.

        Returns
        ----------
        (N,H,W,C) ndarray or list of (H,W,C) ndarray
            The augmented images.

        """
        return images

    def augment_heatmaps(self, heatmaps, parents=None, hooks=None):
        """Augment a batch of heatmaps.

        Parameters
        ----------
        heatmaps : imgaug.augmentables.heatmaps.HeatmapsOnImage or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmap(s) to augment. Either a single heatmap or a list of
            heatmaps.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``.
            It is set automatically for child augmenters.

        hooks : None or imaug.imgaug.HooksHeatmaps, optional
            :class:`~imgaug.imgaug.HooksHeatmaps` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage or list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Corresponding augmented heatmap(s).

        """
        return self.augment_batch_(
            UnnormalizedBatch(heatmaps=heatmaps), parents=parents, hooks=hooks
        ).heatmaps_aug

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        """Augment a batch of heatmaps in-place.

        This is the internal version of :func:`Augmenter.augment_heatmaps`.
        It is called from :func:`Augmenter.augment_heatmaps` and should
        usually not be called directly.
        This method may augment heatmaps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        heatmaps : list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps to augment. They may be changed in-place.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_heatmaps`.

        hooks : imgaug.imgaug.HooksHeatmaps or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_heatmaps`.

        Returns
        ----------
        images : list of imgaug.augmentables.heatmaps.HeatmapsOnImage
            The augmented heatmaps.

        """
        return heatmaps

    def augment_segmentation_maps(self, segmaps, parents=None, hooks=None):
        """Augment a batch of segmentation maps.

        Parameters
        ----------
        segmaps : imgaug.augmentables.segmaps.SegmentationMapsOnImage or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation map(s) to augment. Either a single segmentation map
            or a list of segmentation maps.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.HooksHeatmaps, optional
            :class:`~imgaug.imgaug.HooksHeatmaps` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Corresponding augmented segmentation map(s).

        """
        return self.augment_batch_(
            UnnormalizedBatch(segmentation_maps=segmaps),
            parents=parents,
            hooks=hooks
        ).segmentation_maps_aug

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        """Augment a batch of segmentation in-place.

        This is the internal version of
        :func:`Augmenter.augment_segmentation_maps`.
        It is called from :func:`Augmenter.augment_segmentation_maps` and
        should usually not be called directly.
        This method may augment segmentation maps in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        segmaps : list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Segmentation maps to augment. They may be changed in-place.

        parents : list of imgaug.augmenters.meta.Augmenter
            See
            :func:`~imgaug.augmenters.meta.Augmenter.augment_segmentation_maps`.

        hooks : imgaug.imgaug.HooksHeatmaps or None
            See
            :func:`~imgaug.augmenters.meta.Augmenter.augment_segmentation_maps`.

        Returns
        ----------
        images : list of imgaug.augmentables.segmaps.SegmentationMapsOnImage
            The augmented segmentation maps.

        """
        return segmaps

    def augment_keypoints(self, keypoints_on_images, parents=None, hooks=None):
        """Augment a batch of keypoints/landmarks.

        This is the corresponding function to :func:`Augmenter.augment_images`,
        just for keypoints/landmarks (i.e. points on images).
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_keypoints()`` with the corresponding list of keypoints on
        these images, e.g. ``augment_keypoints([Ak, Bk, Ck])``, where ``Ak``
        are the keypoints on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding keypoints,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.kps import Keypoint
        >>> from imgaug.augmentables.kps import KeypointsOnImage
        >>> A = B = C = np.zeros((10, 10), dtype=np.uint8)
        >>> Ak = Bk = Ck = KeypointsOnImage([Keypoint(2, 2)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> kps_aug = seq_det.augment_keypoints([Ak, Bk, Ck])

        Otherwise, different random values will be sampled for the image
        and keypoint augmentations, resulting in different augmentations (e.g.
        images might be rotated by ``30deg`` and keypoints by ``-10deg``).
        Also make sure to call :func:`Augmenter.to_deterministic` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        keypoints_on_images : imgaug.augmentables.kps.KeypointsOnImage or list of imgaug.augmentables.kps.KeypointsOnImage
            The keypoints/landmarks to augment.
            Either a single instance of
            :class:`~imgaug.augmentables.kps.KeypointsOnImage` or a list of
            such instances. Each instance must contain the keypoints of a
            single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`~imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.kps.KeypointsOnImage or list of imgaug.augmentables.kps.KeypointsOnImage
            Augmented keypoints.

        """
        return self.augment_batch_(
            UnnormalizedBatch(keypoints=keypoints_on_images),
            parents=parents,
            hooks=hooks
        ).keypoints_aug

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        """Augment a batch of keypoints in-place.

        This is the internal version of :func:`Augmenter.augment_keypoints`.
        It is called from :func:`Augmenter.augment_keypoints` and should
        usually not be called directly.
        This method may transform the keypoints in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        keypoints_on_images : list of imgaug.augmentables.kps.KeypointsOnImage
            Keypoints to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_keypoints`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_keypoints`.

        Returns
        ----------
        list of imgaug.augmentables.kps.KeypointsOnImage
            The augmented keypoints.

        """
        return keypoints_on_images

    def augment_bounding_boxes(self, bounding_boxes_on_images, parents=None,
                               hooks=None):
        """Augment a batch of bounding boxes.

        This is the corresponding function to
        :func:`Augmenter.augment_images`, just for bounding boxes.
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_bounding_boxes()`` with the corresponding list of bounding
        boxes on these images, e.g.
        ``augment_bounding_boxes([Abb, Bbb, Cbb])``, where ``Abb`` are the
        bounding boxes on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding bounding boxes,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.bbs import BoundingBox
        >>> from imgaug.augmentables.bbs import BoundingBoxesOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Abb = Bbb = Cbb = BoundingBoxesOnImage([
        >>>     BoundingBox(1, 1, 9, 9)], (10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> bbs_aug = seq_det.augment_bounding_boxes([Abb, Bbb, Cbb])

        Otherwise, different random values will be sampled for the image
        and bounding box augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and bounding boxes by
        ``-10deg``). Also make sure to call :func:`Augmenter.to_deterministic`
        again for each new batch, otherwise you would augment all batches in
        the same way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        bounding_boxes_on_images : imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            The bounding boxes to augment.
            Either a single instance of
            :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage` or a list of
            such instances, with each one of them containing the bounding
            boxes of a single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`~imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            Augmented bounding boxes.

        """
        return self.augment_batch_(
            UnnormalizedBatch(bounding_boxes=bounding_boxes_on_images),
            parents=parents,
            hooks=hooks
        ).bounding_boxes_aug

    def augment_polygons(self, polygons_on_images, parents=None, hooks=None):
        """Augment a batch of polygons.

        This is the corresponding function to :func:`Augmenter.augment_images`,
        just for polygons.
        Usually you will want to call :func:`Augmenter.augment_images`` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_polygons()`` with the corresponding list of polygons on these
        images, e.g. ``augment_polygons([A_poly, B_poly, C_poly])``, where
        ``A_poly`` are the polygons on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding polygons,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.polys import Polygon, PolygonsOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> Apoly = Bpoly = Cpoly = PolygonsOnImage(
        >>>     [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        >>>     shape=(10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> polys_aug = seq_det.augment_polygons([Apoly, Bpoly, Cpoly])

        Otherwise, different random values will be sampled for the image
        and polygon augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and polygons by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        polygons_on_images : imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.polys.PolygonsOnImage
            The polygons to augment.
            Either a single instance of
            :class:`~imgaug.augmentables.polys.PolygonsOnImage` or a list of
            such instances, with each one of them containing the polygons of
            a single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as
            ``None``. It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`~imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.polys.PolygonsOnImage
            Augmented polygons.

        """
        return self.augment_batch_(
            UnnormalizedBatch(polygons=polygons_on_images),
            parents=parents,
            hooks=hooks
        ).polygons_aug

    def augment_line_strings(self, line_strings_on_images, parents=None,
                             hooks=None):
        """Augment a batch of line strings.

        This is the corresponding function to
        :func:`Augmenter.augment_images``, just for line strings.
        Usually you will want to call :func:`Augmenter.augment_images` with
        a list of images, e.g. ``augment_images([A, B, C])`` and then
        ``augment_line_strings()`` with the corresponding list of line
        strings on these images, e.g.
        ``augment_line_strings([A_line, B_line, C_line])``, where ``A_line``
        are the line strings on image ``A``.

        Make sure to first convert the augmenter(s) to deterministic states
        before augmenting images and their corresponding line strings,
        e.g. by

        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.lines import LineString
        >>> from imgaug.augmentables.lines import LineStringsOnImage
        >>> A = B = C = np.ones((10, 10), dtype=np.uint8)
        >>> A_line = B_line = C_line = LineStringsOnImage(
        >>>     [LineString([(0, 0), (1, 0), (1, 1), (0, 1)])],
        >>>     shape=(10, 10))
        >>> seq = iaa.Fliplr(0.5)
        >>> seq_det = seq.to_deterministic()
        >>> imgs_aug = seq_det.augment_images([A, B, C])
        >>> lines_aug = seq_det.augment_line_strings([A_line, B_line, C_line])

        Otherwise, different random values will be sampled for the image
        and line string augmentations, resulting in different augmentations
        (e.g. images might be rotated by ``30deg`` and line strings by
        ``-10deg``). Also make sure to call ``to_deterministic()`` again for
        each new batch, otherwise you would augment all batches in the same
        way.

        Note that there is also :func:`Augmenter.augment`, which automatically
        handles the random state alignment.

        Parameters
        ----------
        line_strings_on_images : imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage
            The line strings to augment.
            Either a single instance of
            :class:`~imgaug.augmentables.lines.LineStringsOnImage` or a list of
            such instances, with each one of them containing the line strings
            of a single image.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            Parent augmenters that have previously been called before the
            call to this function. Usually you can leave this parameter as None.
            It is set automatically for child augmenters.

        hooks : None or imgaug.imgaug.HooksKeypoints, optional
            :class:`~imgaug.imgaug.HooksKeypoints` object to dynamically
            interfere with the augmentation process.

        Returns
        -------
        imgaug.augmentables.lines.LineStringsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage
            Augmented line strings.

        """
        return self.augment_batch_(
            UnnormalizedBatch(line_strings=line_strings_on_images),
            parents=parents,
            hooks=hooks
        ).line_strings_aug

    def _augment_bounding_boxes(self, bounding_boxes_on_images, random_state,
                                parents, hooks):
        """Augment a batch of bounding boxes on images in-place.

        This is the internal version of
        :func:`Augmenter.augment_bounding_boxes`.
        It is called from :func:`Augmenter.augment_bounding_boxes` and should
        usually not be called directly.
        This method may transform the bounding boxes in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Added in 0.4.0.

        Parameters
        ----------
        bounding_boxes_on_images : list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_bounding_boxes`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_bounding_boxes`.

        Returns
        -------
        list of imgaug.augmentables.bbs.BoundingBoxesOnImage
            The augmented bounding boxes.

        """
        return bounding_boxes_on_images

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        """Augment a batch of polygons on images in-place.

        This is the internal version of :func:`Augmenter.augment_polygons`.
        It is called from :func:`Augmenter.augment_polygons` and should
        usually not be called directly.
        This method may transform the polygons in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        polygons_on_images : list of imgaug.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        -------
        list of imgaug.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        return polygons_on_images

    def _augment_line_strings(self, line_strings_on_images, random_state,
                              parents, hooks):
        """Augment a batch of line strings in-place.

        This is the internal version of
        :func:`Augmenter.augment_line_strings`.
        It is called from :func:`Augmenter.augment_line_strings` and should
        usually not be called directly.
        This method may transform the line strings in-place.
        This method does not have to care about determinism or the
        Augmenter instance's ``random_state`` variable. The parameter
        ``random_state`` takes care of both of these.

        .. note::

            This method exists mostly for legacy-support.
            Overwriting :func:`~imgaug.augmenters.meta.Augmenter._augment_batch`
            is now the preferred way of implementing custom augmentation
            routines.

        Parameters
        ----------
        line_strings_on_images : list of imgaug.augmentables.lines.LineStringsOnImage
            Line strings to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_line_strings`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_line_strings`.

        Returns
        -------
        list of imgaug.augmentables.lines.LineStringsOnImage
            The augmented line strings.

        """
        return line_strings_on_images

    def _augment_bounding_boxes_as_keypoints(self, bounding_boxes_on_images,
                                             random_state, parents, hooks):
        """
        Augment BBs by applying keypoint augmentation to their corners.

        Added in 0.4.0.

        Parameters
        ----------
        bounding_boxes_on_images : list of imgaug.augmentables.bbs.BoundingBoxesOnImages or imgaug.augmentables.bbs.BoundingBoxesOnImages
            Bounding boxes to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        -------
        list of imgaug.augmentables.bbs.BoundingBoxesOnImage or imgaug.augmentables.bbs.BoundingBoxesOnImage
            The augmented bounding boxes.

        """
        return self._augment_cbaois_as_keypoints(bounding_boxes_on_images,
                                                 random_state=random_state,
                                                 parents=parents,
                                                 hooks=hooks)

    def _augment_polygons_as_keypoints(self, polygons_on_images, random_state,
                                       parents, hooks, recoverer=None):
        """
        Augment polygons by applying keypoint augmentation to their vertices.

        .. warning::

            This method calls
            :func:`~imgaug.augmenters.meta.Augmenter._augment_keypoints` and
            expects it to do keypoint augmentation. The default for that
            method is to do nothing. It must therefore be overwritten,
            otherwise the polygon augmentation will also do nothing.

        Parameters
        ----------
        polygons_on_images : list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        recoverer : None or imgaug.augmentables.polys._ConcavePolygonRecoverer
            An instance used to repair invalid polygons after augmentation.
            Must offer the method
            ``recover_from(new_exterior, old_polygon, random_state=0)``.
            If ``None`` then invalid polygons are not repaired.

        Returns
        -------
        list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        func = functools.partial(self._augment_keypoints,
                                 random_state=random_state,
                                 parents=parents,
                                 hooks=hooks)

        return self._apply_to_polygons_as_keypoints(polygons_on_images, func,
                                                    recoverer, random_state)

    def _augment_line_strings_as_keypoints(self, line_strings_on_images,
                                           random_state, parents, hooks):
        """
        Augment BBs by applying keypoint augmentation to their corners.

        Parameters
        ----------
        line_strings_on_images : list of imgaug.augmentables.lines.LineStringsOnImages or imgaug.augmentables.lines.LineStringsOnImages
            Line strings to augment. They may be changed in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_polygons`.

        Returns
        -------
        list of imgaug.augmentables.lines.LineStringsOnImages or imgaug.augmentables.lines.LineStringsOnImages
            The augmented line strings.

        """
        return self._augment_cbaois_as_keypoints(line_strings_on_images,
                                                 random_state=random_state,
                                                 parents=parents,
                                                 hooks=hooks)

    def _augment_cbaois_as_keypoints(
            self, cbaois, random_state, parents, hooks):
        """
        Augment bounding boxes by applying KP augmentation to their corners.

        Added in 0.4.0.

        Parameters
        ----------
        cbaois : list of imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage or imgaug.augmentables.bbs.BoundingBoxesOnImage or imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.lines.LineStringsOnImage
            Coordinate-based augmentables to augment. They may be changed
            in-place.

        random_state : imgaug.random.RNG
            The random state to use for all sampling tasks during the
            augmentation.

        parents : list of imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_batch`.

        hooks : imgaug.imgaug.HooksKeypoints or None
            See :func:`~imgaug.augmenters.meta.Augmenter.augment_batch`.

        Returns
        -------
        list of imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage or imgaug.augmentables.bbs.BoundingBoxesOnImage or imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.lines.LineStringsOnImage
            The augmented coordinate-based augmentables.

        """
        func = functools.partial(self._augment_keypoints,
                                 random_state=random_state,
                                 parents=parents,
                                 hooks=hooks)
        return self._apply_to_cbaois_as_keypoints(cbaois, func)

    @classmethod
    def _apply_to_polygons_as_keypoints(cls, polygons_on_images, func,
                                        recoverer=None, random_state=None):
        """
        Apply a callback to polygons in keypoint-representation.

        Added in 0.4.0.

        Parameters
        ----------
        polygons_on_images : list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
            Polygons to augment. They may be changed in-place.

        func : callable
            The function to apply. Receives a list of
            :class:`~imgaug.augmentables.kps.KeypointsOnImage` instances as its
            only parameter.

        recoverer : None or imgaug.augmentables.polys._ConcavePolygonRecoverer
            An instance used to repair invalid polygons after augmentation.
            Must offer the method
            ``recover_from(new_exterior, old_polygon, random_state=0)``.
            If ``None`` then invalid polygons are not repaired.

        random_state : None or imgaug.random.RNG
            The random state to use for the recoverer.

        Returns
        -------
        list of imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.polys.PolygonsOnImage
            The augmented polygons.

        """
        from ..augmentables.polys import recover_psois_

        psois_orig = None
        if recoverer is not None:
            if isinstance(polygons_on_images, list):
                psois_orig = [psoi.deepcopy() for psoi in polygons_on_images]
            else:
                psois_orig = polygons_on_images.deepcopy()

        psois = cls._apply_to_cbaois_as_keypoints(polygons_on_images, func)

        if recoverer is None:
            return psois

        # Its not really necessary to create an RNG copy for the recoverer
        # here, as the augmentation of the polygons is already finished and
        # used the same samples as the image augmentation. The recoverer might
        # advance the RNG state, but the next call to e.g. augment() will then
        # still use the same (advanced) RNG state for images and polygons.
        # We copy here anyways as it seems cleaner.
        random_state_recoverer = (random_state.copy()
                                  if random_state is not None else None)
        psois = recover_psois_(psois, psois_orig, recoverer,
                               random_state_recoverer)

        return psois

    @classmethod
    def _apply_to_cbaois_as_keypoints(cls, cbaois, func):
        """
        Augment bounding boxes by applying KP augmentation to their corners.

        Added in 0.4.0.

        Parameters
        ----------
        cbaois : list of imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage or imgaug.augmentables.bbs.BoundingBoxesOnImage or imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.lines.LineStringsOnImage
            Coordinate-based augmentables to augment. They may be changed
            in-place.

        func : callable
            The function to apply. Receives a list of
            :class:`~imgaug.augmentables.kps.KeypointsOnImage` instances as its
            only parameter.

        Returns
        -------
        list of imgaug.augmentables.bbs.BoundingBoxesOnImage or list of imgaug.augmentables.polys.PolygonsOnImage or list of imgaug.augmentables.lines.LineStringsOnImage or imgaug.augmentables.bbs.BoundingBoxesOnImage or imgaug.augmentables.polys.PolygonsOnImage or imgaug.augmentables.lines.LineStringsOnImage
            The augmented coordinate-based augmentables.

        """
        from ..augmentables.utils import (convert_cbaois_to_kpsois,
                                          invert_convert_cbaois_to_kpsois_)

        kpsois = convert_cbaois_to_kpsois(cbaois)
        kpsois_aug = func(kpsois)
        return invert_convert_cbaois_to_kpsois_(cbaois, kpsois_aug)

    def augment(self, return_batch=False, hooks=None, **kwargs):
        """Augment a batch.

        This method is a wrapper around
        :class:`~imgaug.augmentables.batches.UnnormalizedBatch` and
        :func:`~imgaug.augmenters.meta.Augmenter.augment_batch`. Hence, it
        supports the same datatypes as
        :class:`~imgaug.augmentables.batches.UnnormalizedBatch`.

        If `return_batch` was set to ``False`` (the default), the method will
        return a tuple of augmentables. It will return the same types of
        augmentables (but in augmented form) as input into the method. This
        behaviour is partly specific to the python version:

        * In **python 3.6+** (if ``return_batch=False``):

            * Any number of augmentables may be provided as input.
            * None of the provided named arguments *has to be* `image` or
              `images` (but of coarse you *may* provide them).
            * The return order matches the order of the named arguments, e.g.
              ``x_aug, y_aug, z_aug = augment(X=x, Y=y, Z=z)``.

        * In **python <3.6** (if ``return_batch=False``):

            * One or two augmentables may be used as input, not more than that.
            * One of the input arguments has to be `image` or `images`.
            * The augmented images are *always* returned first, independent
              of the input argument order, e.g.
              ``a_aug, b_aug = augment(b=b, images=a)``. This also means
              that the output of the function can only be one of the
              following three cases: a batch, list/array of images,
              tuple of images and something (like images + segmentation maps).

        If `return_batch` was set to ``True``, an instance of
        :class:`~imgaug.augmentables.batches.UnnormalizedBatch` will be
        returned. The output is the same for all python version and any
        number or combination of augmentables may be provided.

        So, to keep code downward compatible for python <3.6, use one of the
        following three options:

          * Use ``batch = augment(images=X, ..., return_batch=True)``.
          * Call ``images = augment(images=X)``.
          * Call ``images, other = augment(images=X, <something_else>=Y)``.

        All augmentables must be provided as named arguments.
        E.g. ``augment(<array>)`` will crash, but ``augment(images=<array>)``
        will work.

        Parameters
        ----------
        image : None or (H,W,C) ndarray or (H,W) ndarray, optional
            The image to augment. Only this or `images` can be set, not both.
            If `return_batch` is ``False`` and the python version is below 3.6,
            either this or `images` **must** be provided.

        images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or iterable of (H,W,C) ndarray or iterable of (H,W) ndarray, optional
            The images to augment. Only this or `image` can be set, not both.
            If `return_batch` is ``False`` and the python version is below 3.6,
            either this or `image` **must** be provided.

        heatmaps : None or (N,H,W,C) ndarray or imgaug.augmentables.heatmaps.HeatmapsOnImage or iterable of (H,W,C) ndarray or iterable of imgaug.augmentables.heatmaps.HeatmapsOnImage, optional
            The heatmaps to augment.
            If anything else than
            :class:`~imgaug.augmentables.heatmaps.HeatmapsOnImage`, then the
            number of heatmaps must match the number of images provided via
            parameter `images`. The number is contained either in ``N`` or the
            first iterable's size.

        segmentation_maps : None or (N,H,W) ndarray or imgaug.augmentables.segmaps.SegmentationMapsOnImage or iterable of (H,W) ndarray or iterable of imgaug.augmentables.segmaps.SegmentationMapsOnImage, optional
            The segmentation maps to augment.
            If anything else than
            :class:`~imgaug.augmentables.segmaps.SegmentationMapsOnImage`, then
            the number of segmaps must match the number of images provided via
            parameter `images`. The number is contained either in ``N`` or the
            first iterable's size.

        keypoints : None or list of (N,K,2) ndarray or tuple of number or imgaug.augmentables.kps.Keypoint or iterable of (K,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.kps.KeypointOnImage or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint, optional
            The keypoints to augment.
            If a tuple (or iterable(s) of tuple), then iterpreted as ``(x,y)``
            coordinates and must hence contain two numbers.
            A single tuple represents a single coordinate on one image, an
            iterable of tuples the coordinates on one image and an iterable of
            iterable of tuples the coordinates on several images. Analogous if
            :class:`~imgaug.augmentables.kps.Keypoint` instances are used
            instead of tuples.
            If an ndarray, then ``N`` denotes the number of images and ``K``
            the number of keypoints on each image.
            If anything else than
            :class:`~imgaug.augmentables.kps.KeypointsOnImage` is provided, then
            the number of keypoint groups must match the number of images
            provided via parameter `images`. The number is contained e.g. in
            ``N`` or in case of "iterable of iterable of tuples" in the first
            iterable's size.

        bounding_boxes : None or (N,B,4) ndarray or tuple of number or imgaug.augmentables.bbs.BoundingBox or imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of (B,4) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.bbs.BoundingBox or iterable of imgaug.augmentables.bbs.BoundingBoxesOnImage or iterable of iterable of tuple of number or iterable of iterable imgaug.augmentables.bbs.BoundingBox, optional
            The bounding boxes to augment.
            This is analogous to the `keypoints` parameter. However, each
            tuple -- and also the last index in case of arrays -- has size
            ``4``, denoting the bounding box coordinates ``x1``, ``y1``,
            ``x2`` and ``y2``.

        polygons : None or (N,#polys,#points,2) ndarray or imgaug.augmentables.polys.Polygon or imgaug.augmentables.polys.PolygonsOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.polys.Polygon or iterable of imgaug.augmentables.polys.PolygonsOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.polys.Polygon or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint, optional
            The polygons to augment.
            This is similar to the `keypoints` parameter. However, each polygon
            may be made up of several ``(x,y) ``coordinates (three or more are
            required for valid polygons).
            The following datatypes will be interpreted as a single polygon on
            a single image:

              * ``imgaug.augmentables.polys.Polygon``
              * ``iterable of tuple of number``
              * ``iterable of imgaug.augmentables.kps.Keypoint``

            The following datatypes will be interpreted as multiple polygons
            on a single image:

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

        line_strings : None or (N,#lines,#points,2) ndarray or imgaug.augmentables.lines.LineString or imgaug.augmentables.lines.LineStringOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug.augmentables.kps.Keypoint or iterable of imgaug.augmentables.lines.LineString or iterable of imgaug.augmentables.lines.LineStringOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug.augmentables.kps.Keypoint or iterable of iterable of imgaug.augmentables.lines.LineString or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug.augmentables.kps.Keypoint, optional
            The line strings to augment.
            See `polygons`, which behaves similarly.

        return_batch : bool, optional
            Whether to return an instance of
            :class:`~imgaug.augmentables.batches.UnnormalizedBatch`. If the
            python version is below 3.6 and more than two augmentables were
            provided (e.g. images, keypoints and polygons), then this must be
            set to ``True``. Otherwise an error will be raised.

        hooks : None or imgaug.imgaug.HooksImages, optional
            Hooks object to dynamically interfere with the augmentation process.

        Returns
        -------
        tuple or imgaug.augmentables.batches.UnnormalizedBatch
            If `return_batch` was set to ``True``, a instance of
            ``UnnormalizedBatch`` will be returned.
            If `return_batch` was set to ``False``, a tuple of augmentables
            will be returned, e.g. ``(augmented images, augmented keypoints)``.
            The datatypes match the input datatypes of the corresponding named
            arguments. In python <3.6, augmented images are always the first
            entry in the returned tuple. In python 3.6+ the order matches the
            order of the named arguments.

        Examples
        --------
        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> aug = iaa.Affine(rotate=(-25, 25))
        >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
        >>> keypoints = [(10, 20), (30, 32)]  # (x,y) coordinates
        >>> images_aug, keypoints_aug = aug.augment(
        >>>     image=image, keypoints=keypoints)

        Create a single image and a set of two keypoints on it, then
        augment both by applying a random rotation between ``-25`` deg and
        ``+25`` deg. The sampled rotation value is automatically aligned
        between image and keypoints. Note that in python <3.6, augmented
        images will always be returned first, independent of the order of
        the named input arguments. So
        ``keypoints_aug, images_aug = aug.augment(keypoints=keypoints,
        image=image)`` would **not** be correct (but in python 3.6+ it would
        be).

        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.bbs import BoundingBox
        >>> aug = iaa.Affine(rotate=(-25, 25))
        >>> images = [np.zeros((64, 64, 3), dtype=np.uint8),
        >>>           np.zeros((32, 32, 3), dtype=np.uint8)]
        >>> keypoints = [[(10, 20), (30, 32)],  # KPs on first image
        >>>              [(22, 10), (12, 14)]]  # KPs on second image
        >>> bbs = [
        >>>           [BoundingBox(x1=5, y1=5, x2=50, y2=45)],
        >>>           [BoundingBox(x1=4, y1=6, x2=10, y2=15),
        >>>            BoundingBox(x1=8, y1=9, x2=16, y2=30)]
        >>>       ]  # one BB on first image, two BBs on second image
        >>> batch_aug = aug.augment(
        >>>     images=images, keypoints=keypoints, bounding_boxes=bbs,
        >>>     return_batch=True)

        Create two images of size ``64x64`` and ``32x32``, two sets of
        keypoints (each containing two keypoints) and two sets of bounding
        boxes (the first containing one bounding box, the second two bounding
        boxes). These augmentables are then augmented by applying random
        rotations between ``-25`` deg and ``+25`` deg to them. The rotation
        values are sampled by image and aligned between all augmentables on
        the same image. The method finally returns an instance of
        :class:`~imgaug.augmentables.batches.UnnormalizedBatch` from which the
        augmented data can be retrieved via ``batch_aug.images_aug``,
        ``batch_aug.keypoints_aug``, and ``batch_aug.bounding_boxes_aug``.
        In python 3.6+, `return_batch` can be kept at ``False`` and the
        augmented data can be retrieved as
        ``images_aug, keypoints_aug, bbs_aug = augment(...)``.

        """
        assert ia.is_single_bool(return_batch), (
            "Expected boolean as argument for 'return_batch', got type %s. "
            "Call augment() only with named arguments, e.g. "
            "augment(images=<array>)." % (str(type(return_batch)),))

        expected_keys = ["images", "heatmaps", "segmentation_maps",
                         "keypoints", "bounding_boxes", "polygons",
                         "line_strings"]
        expected_keys_call = ["image"] + expected_keys

        # at least one augmentable provided?
        assert any([key in kwargs for key in expected_keys_call]), (
            "Expected augment() to be called with one of the following named "
            "arguments: %s. Got none of these." % (
                ", ".join(expected_keys_call),))

        # all keys in kwargs actually known?
        unknown_args = [key for key in kwargs if key not in expected_keys_call]
        assert len(unknown_args) == 0, (
            "Got the following unknown keyword argument(s) in augment(): %s" % (
                ", ".join(unknown_args)
            ))

        # normalize image=... input to images=...
        # this is not done by Batch.to_normalized_batch()
        if "image" in kwargs:
            assert "images" not in kwargs, (
                "You may only provide the argument 'image' OR 'images' to "
                "augment(), not both of them.")
            images = [kwargs["image"]]
            iabase._warn_on_suspicious_single_image_shape(images[0])
        else:
            images = kwargs.get("images", None)
            iabase._warn_on_suspicious_multi_image_shapes(images)

        # Decide whether to return the final tuple in the order of the kwargs
        # keys or the default order based on python version. Only 3.6+ uses
        # an ordered dict implementation for kwargs.
        order = "standard"
        nb_keys = len(list(kwargs.keys()))
        vinfo = sys.version_info
        is_py36_or_newer = vinfo[0] > 3 or (vinfo[0] == 3 and vinfo[1] >= 6)
        if is_py36_or_newer:
            order = "kwargs_keys"
        elif not return_batch and nb_keys > 2:
            raise ValueError(
                "Requested more than two outputs in augment(), but detected "
                "python version is below 3.6. More than two outputs are only "
                "supported for 3.6+ as earlier python versions offer no way "
                "to retrieve the order of the provided named arguments. To "
                "still use more than two outputs, add 'return_batch=True' as "
                "an argument and retrieve the outputs manually from the "
                "returned UnnormalizedBatch instance, e.g. via "
                "'batch.images_aug' to get augmented images."
            )
        elif not return_batch and nb_keys == 2 and images is None:
            raise ValueError(
                "Requested two outputs from augment() that were not 'images', "
                "but detected python version is below 3.6. For security "
                "reasons, only single-output requests or requests with two "
                "outputs of which one is 'images' are allowed in <3.6. "
                "'images' will then always be returned first. If you don't "
                "want this, use 'return_batch=True' mode in augment(), which "
                "returns a single UnnormalizedBatch instance instead and "
                "supports any combination of outputs."
            )

        # augment batch
        batch = UnnormalizedBatch(
            images=images,
            heatmaps=kwargs.get("heatmaps", None),
            segmentation_maps=kwargs.get("segmentation_maps", None),
            keypoints=kwargs.get("keypoints", None),
            bounding_boxes=kwargs.get("bounding_boxes", None),
            polygons=kwargs.get("polygons", None),
            line_strings=kwargs.get("line_strings", None)
        )

        batch_aug = self.augment_batch_(batch, hooks=hooks)

        # return either batch or tuple of augmentables, depending on what
        # was requested by user
        if return_batch:
            return batch_aug

        result = []
        if order == "kwargs_keys":
            for key in kwargs:
                if key == "image":
                    attr = getattr(batch_aug, "images_aug")
                    result.append(attr[0])
                else:
                    result.append(getattr(batch_aug, "%s_aug" % (key,)))
        else:
            for key in expected_keys:
                if key == "images" and "image" in kwargs:
                    attr = getattr(batch_aug, "images_aug")
                    result.append(attr[0])
                elif key in kwargs:
                    result.append(getattr(batch_aug, "%s_aug" % (key,)))

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def __call__(self, *args, **kwargs):
        """Alias for :func:`~imgaug.augmenters.meta.Augmenter.augment`."""
        return self.augment(*args, **kwargs)

    def pool(self, processes=None, maxtasksperchild=None, seed=None):
        """Create a pool used for multicore augmentation.

        Parameters
        ----------
        processes : None or int, optional
            Same as in :func:`~imgaug.multicore.Pool.__init__`.
            The number of background workers. If ``None``, the number of the
            machine's CPU cores will be used (this counts hyperthreads as CPU
            cores). If this is set to a negative value ``p``, then
            ``P - abs(p)`` will be used, where ``P`` is the number of CPU
            cores. E.g. ``-1`` would use all cores except one (this is useful
            to e.g. reserve one core to feed batches to the GPU).

        maxtasksperchild : None or int, optional
            Same as for :func:`~imgaug.multicore.Pool.__init__`.
            The number of tasks done per worker process before the process
            is killed and restarted. If ``None``, worker processes will not
            be automatically restarted.

        seed : None or int, optional
            Same as for :func:`~imgaug.multicore.Pool.__init__`.
            The seed to use for child processes. If ``None``, a random seed
            will be used.

        Returns
        -------
        imgaug.multicore.Pool
            Pool for multicore augmentation.

        Examples
        --------
        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.batches import Batch
        >>>
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> batches = [Batch(images=np.copy(images)) for _ in range(100)]
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.map_batches(batches, chunksize=8)
        >>> print(np.sum(batches_aug[0].images_aug[0]))
        49152

        Create ``100`` batches of empty images. Each batch contains
        ``16`` images of size ``128x128``. The batches are then augmented on
        all CPU cores except one (``processes=-1``). After augmentation, the
        sum of pixel values from the first augmented image is printed.

        >>> import numpy as np
        >>> import imgaug as ia
        >>> import imgaug.augmenters as iaa
        >>> from imgaug.augmentables.batches import Batch
        >>>
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> def generate_batches():
        >>>     for _ in range(100):
        >>>         yield Batch(images=np.copy(images))
        >>>
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.imap_batches(generate_batches(), chunksize=8)
        >>>     batch_aug = next(batches_aug)
        >>>     print(np.sum(batch_aug.images_aug[0]))
        49152

        Same as above. This time, a generator is used to generate batches
        of images. Again, the first augmented image's sum of pixels is printed.

        """
        import imgaug.multicore as multicore
        return multicore.Pool(self, processes=processes,
                              maxtasksperchild=maxtasksperchild, seed=seed)

    # TODO most of the code of this function could be replaced with
    #      ia.draw_grid()
    # TODO add parameter for handling multiple images ((a) next to each other
    #      in each row or (b) multiply row count by number of images and put
    #      each one in a new row)
    # TODO "images" parameter deviates from augment_images (3d array is here
    #      treated as one 3d image, in augment_images as (N, H, W))
    # TODO according to the docstring, this can handle (H,W) images, but not
    #      (H,W,1)
    def draw_grid(self, images, rows, cols):
        """Augment images and draw the results as a single grid-like image.

        This method applies this augmenter to the provided images and returns
        a grid image of the results. Each cell in the grid contains a single
        augmented version of an input image.

        If multiple input images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for ``images = [A, B]``, ``rows=2``, ``cols=3``::

            A A A
            B B B
            A A A
            B B B

        for ``images = [A]``, ``rows=2``, ``cols=3``::

            A A A
            A A A

        Parameters
        -------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images to augment and draw in the grid.
            If a list, then each element is expected to have shape ``(H, W)``
            or ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        Returns
        -------
        (Hg, Wg, 3) ndarray
            The generated grid image with augmented versions of the input
            images. Here, ``Hg`` and ``Wg`` reference the output size of the
            grid, and *not* the sizes of the input images.

        """
        if ia.is_np_array(images):
            if len(images.shape) == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif len(images.shape) == 3:
                images = [images]
            elif len(images.shape) == 2:
                images = [images[:, :, np.newaxis]]
            else:
                raise Exception(
                    "Unexpected images shape, expected 2-, 3- or "
                    "4-dimensional array, got shape %s." % (images.shape,))
        else:
            assert isinstance(images, list), (
                "Expected 'images' to be an ndarray or list of ndarrays. "
                "Got %s." % (type(images),))
            for i, image in enumerate(images):
                if len(image.shape) == 3:
                    continue
                if len(image.shape) == 2:
                    images[i] = image[:, :, np.newaxis]
                else:
                    raise Exception(
                        "Unexpected image shape at index %d, expected 2- or "
                        "3-dimensional array, got shape %s." % (
                            i, image.shape,))

        det = self if self.deterministic else self.to_deterministic()
        augs = []
        for image in images:
            augs.append(det.augment_images([image] * (rows * cols)))

        augs_flat = list(itertools.chain(*augs))
        cell_height = max([image.shape[0] for image in augs_flat])
        cell_width = max([image.shape[1] for image in augs_flat])
        width = cell_width * cols
        height = cell_height * (rows * len(images))
        grid = np.zeros((height, width, 3), dtype=augs[0][0].dtype)
        for row_idx in range(rows):
            for img_idx, image in enumerate(images):
                for col_idx in range(cols):
                    image_aug = augs[img_idx][(row_idx * cols) + col_idx]
                    cell_y1 = cell_height * (row_idx * len(images) + img_idx)
                    cell_y2 = cell_y1 + image_aug.shape[0]
                    cell_x1 = cell_width * col_idx
                    cell_x2 = cell_x1 + image_aug.shape[1]
                    grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image_aug

        return grid

    # TODO test for 2D images
    # TODO test with C = 1
    def show_grid(self, images, rows, cols):
        """Augment images and plot the results as a single grid-like image.

        This calls :func:`~imgaug.augmenters.meta.Augmenter.draw_grid` and
        simply shows the results. See that method for details.

        Parameters
        ----------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images to augment and draw in the grid.
            If a list, then each element is expected to have shape ``(H, W)``
            or ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        """
        grid = self.draw_grid(images, rows, cols)
        ia.imshow(grid)

    def to_deterministic(self, n=None):
        """Convert this augmenter from a stochastic to a deterministic one.

        A stochastic augmenter samples pseudo-random values for each parameter,
        image and batch.
        A deterministic augmenter also samples new values for each parameter
        and image, but not batch. Instead, for consecutive batches it will
        sample the same values (provided the number of images and their sizes
        don't change).
        From a technical perspective this means that a deterministic augmenter
        starts each batch's augmentation with a random number generator in
        the same state (i.e. same seed), instead of advancing that state from
        batch to batch.

        Using determinism is useful to (a) get the same augmentations for
        two or more image batches (e.g. for stereo cameras), (b) to augment
        images and corresponding data on them (e.g. segmentation maps or
        bounding boxes) in the same way.

        Parameters
        ----------
        n : None or int, optional
            Number of deterministic augmenters to return.
            If ``None`` then only one :class:`~imgaug.augmenters.meta.Augmenter`
            instance will be returned.
            If ``1`` or higher, a list containing ``n``
            :class:`~imgaug.augmenters.meta.Augmenter` instances will be
            returned.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter
            A single Augmenter object if `n` was None,
            otherwise a list of Augmenter objects (even if `n` was ``1``).

        """
        assert n is None or n >= 1, (
            "Expected 'n' to be None or >=1, got %s." % (n,))
        if n is None:
            return self.to_deterministic(1)[0]
        return [self._to_deterministic() for _ in sm.xrange(n)]

    def _to_deterministic(self):
        """Convert this augmenter from a stochastic to a deterministic one.

        Augmenter-specific implementation of
        :func:`~imgaug.augmenters.meta.to_deterministic`. This function is
        expected to return a single new deterministic
        :class:`~imgaug.augmenters.meta.Augmenter` instance of this augmenter.

        Returns
        -------
        det : imgaug.augmenters.meta.Augmenter
            Deterministic variation of this Augmenter object.

        """
        aug = self.copy()

        # This was changed for 0.2.8 from deriving a new random state based on
        # the global random state to deriving it from the augmenter's local
        # random state. This should reduce the risk that re-runs of scripts
        # lead to different results upon small changes somewhere. It also
        # decreases the likelihood of problems when using multiprocessing
        # (the child processes might use the same global random state as the
        # parent process). Note for the latter point that augment_batches()
        # might call to_deterministic() if the batch contains multiply types
        # of augmentables.
        # aug.random_state = iarandom.create_random_rng()
        aug.random_state = self.random_state.derive_rng_()

        aug.deterministic = True
        return aug

    @ia.deprecated("imgaug.augmenters.meta.Augmenter.seed_")
    def reseed(self, random_state=None, deterministic_too=False):
        """Old name of :func:`~imgaug.augmenters.meta.Augmenter.seed_`.

        Deprecated since 0.4.0.

        """
        self.seed_(entropy=random_state, deterministic_too=deterministic_too)

    # TODO mark this as in-place
    def seed_(self, entropy=None, deterministic_too=False):
        """Seed this augmenter and all of its children.

        This method assigns a new random number generator to the
        augmenter and all of its children (if it has any). The new random
        number generator is *derived* from the provided seed or RNG -- or from
        the global random number generator if ``None`` was provided.
        Note that as child RNGs are *derived*, they do not all use the same
        seed.

        If this augmenter or any child augmenter had a random number generator
        that pointed to the global random state, it will automatically be
        replaced with a local random state. This is similar to what
        :func:`~imgaug.augmenters.meta.Augmenter.localize_random_state`
        does.

        This method is useful when augmentations are run in the
        background (i.e. on multiple cores).
        It should be called before sending this
        :class:`~imgaug.augmenters.meta.Augmenter` instance to a
        background worker or once within each worker with different seeds
        (i.e., if ``N`` workers are used, the function should be called
        ``N`` times). Otherwise, all background workers will
        use the same seeds and therefore apply the same augmentations.
        Note that :func:`Augmenter.augment_batches` and :func:`Augmenter.pool`
        already do this automatically.

        Added in 0.4.0.

        Parameters
        ----------
        entropy : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
            A seed or random number generator that is used to derive new
            random number generators for this augmenter and its children.
            If an ``int`` is provided, it will be interpreted as a seed.
            If ``None`` is provided, the global random number generator will
            be used.

        deterministic_too : bool, optional
            Whether to also change the seed of an augmenter ``A``, if ``A``
            is deterministic. This is the case both when this augmenter
            object is ``A`` or one of its children is ``A``.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> aug = iaa.Sequential([
        >>>     iaa.Crop(px=(0, 10)),
        >>>     iaa.Crop(px=(0, 10))
        >>> ])
        >>> aug.seed_(1)

        Seed an augmentation sequence containing two crop operations. Even
        though the same seed was used, the two operations will still sample
        different pixel amounts to crop as the child-specific seed is merely
        derived from the provided seed.

        """
        assert isinstance(deterministic_too, bool), (
            "Expected 'deterministic_too' to be a boolean, got type %s." % (
                deterministic_too))

        if entropy is None:
            random_state = iarandom.RNG.create_pseudo_random_()
        else:
            random_state = iarandom.RNG(entropy)

        if not self.deterministic or deterministic_too:
            # note that derive_rng_() (used below) advances the RNG, so
            # child augmenters get a different RNG state
            self.random_state = random_state.copy()

        for lst in self.get_children_lists():
            for aug in lst:
                aug.seed_(entropy=random_state.derive_rng_(),
                          deterministic_too=deterministic_too)

    def localize_random_state(self, recursive=True):
        """Assign augmenter-specific RNGs to this augmenter and its children.

        See :func:`Augmenter.localize_random_state_` for more details.

        Parameters
        ----------
        recursive : bool, optional
            See
            :func:`~imgaug.augmenters.meta.Augmenter.localize_random_state_`.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Copy of the augmenter and its children, with localized RNGs.

        """
        aug = self.deepcopy()
        aug.localize_random_state_(
            recursive=recursive
        )
        return aug

    # TODO rename random_state -> rng
    def localize_random_state_(self, recursive=True):
        """Assign augmenter-specific RNGs to this augmenter and its children.

        This method iterates over this augmenter and all of its children and
        replaces any pointer to the global RNG with a new local (i.e.
        augmenter-specific) RNG.

        A random number generator (RNG) is used for the sampling of random
        values.
        The global random number generator exists exactly once throughout
        the library and is shared by many augmenters.
        A local RNG (usually) exists within exactly one augmenter and is
        only used by that augmenter.

        Usually there is no need to change global into local RNGs.
        The only noteworthy exceptions are

            * Whenever you want to use determinism (so that the global RNG is
              not accidentally reverted).
            * Whenever you want to copy RNGs from one augmenter to
              another. (Copying the global RNG would usually not be useful.
              Copying the global RNG from augmenter A to B, then executing A
              and then B would result in B's (global) RNG's state having
              already changed because of A's sampling. So the samples of
              A and B would differ.)

        The case of determinism is handled automatically by
        :func:`~imgaug.augmenters.meta.Augmenter.to_deterministic`.
        Only when you copy RNGs (via
        :func:`~imgaug.augmenters.meta.Augmenter.copy_random_state`),
        you need to call this function first.

        Parameters
        ----------
        recursive : bool, optional
            Whether to localize the RNGs of the augmenter's children too.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Returns itself (with localized RNGs).

        """
        if self.random_state.is_global_rng():
            self.random_state = self.random_state.derive_rng_()
        if recursive:
            for lst in self.get_children_lists():
                for child in lst:
                    child.localize_random_state_(recursive=recursive)
        return self

    # TODO adapt random_state -> rng
    def copy_random_state(self, source, recursive=True, matching="position",
                          matching_tolerant=True, copy_determinism=False):
        """Copy the RNGs from a source augmenter sequence.

        Parameters
        ----------
        source : imgaug.augmenters.meta.Augmenter
            See :func:`~imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        recursive : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        matching : {'position', 'name'}, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        matching_tolerant : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        copy_determinism : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.copy_random_state_`.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Copy of the augmenter itself (with copied RNGs).

        """
        aug = self.deepcopy()
        aug.copy_random_state_(
            source,
            recursive=recursive,
            matching=matching,
            matching_tolerant=matching_tolerant,
            copy_determinism=copy_determinism
        )
        return aug

    def copy_random_state_(self, source, recursive=True, matching="position",
                           matching_tolerant=True, copy_determinism=False):
        """Copy the RNGs from a source augmenter sequence (in-place).

        .. note::

            The source augmenters are not allowed to use the global RNG.
            Call
            :func:`~imgaug.augmenters.meta.Augmenter.localize_random_state_`
            once on the source to localize all random states.

        Parameters
        ----------
        source : imgaug.augmenters.meta.Augmenter
            The source augmenter(s) from where to copy the RNG(s).
            The source may have children (e.g. the source can be a
            :class:`~imgaug.augmenters.meta.Sequential`).

        recursive : bool, optional
            Whether to copy the RNGs of the source augmenter *and*
            all of its children (``True``) or just the source
            augmenter (``False``).

        matching : {'position', 'name'}, optional
            Defines the matching mode to use during recursive copy.
            This is used to associate source augmenters with target augmenters.
            If ``position`` then the target and source sequences of augmenters
            are turned into flattened lists and are associated based on
            their list indices. If ``name`` then the target and source
            augmenters are matched based on their names (i.e.
            ``augmenter.name``).

        matching_tolerant : bool, optional
            Whether to use tolerant matching between source and target
            augmenters. If set to ``False``: Name matching will raise an
            exception for any target augmenter which's name does not appear
            among the source augmenters. Position matching will raise an
            exception if source and target augmenter have an unequal number
            of children.

        copy_determinism : bool, optional
            Whether to copy the ``deterministic`` attributes from source to
            target augmenters too.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            The augmenter itself.

        """
        # Note: the target random states are localized, but the source random
        # states don't have to be localized. That means that they can be
        # the global random state. Worse, if copy_random_state() was called,
        # the target random states would have different identities, but
        # same states. If multiple target random states were the global random
        # state, then after deepcopying them, they would all share the same
        # identity that is different to the global random state. I.e., if the
        # state of any random state of them is set in-place, it modifies the
        # state of all other target random states (that were once global),
        # but not the global random state.
        # Summary: Use target = source.copy() here, instead of
        # target.use_state_of_(source).

        source_augs = (
            [source] + source.get_all_children(flat=True)
            if recursive
            else [source])
        target_augs = (
            [self] + self.get_all_children(flat=True)
            if recursive
            else [self])

        global_rs_exc_msg = (
            "You called copy_random_state_() with a source that uses global "
            "RNGs. Call localize_random_state_() on the source "
            "first or initialize your augmenters with local random states, "
            "e.g. via Dropout(..., random_state=1234).")

        if matching == "name":
            source_augs_dict = {aug.name: aug for aug in source_augs}
            target_augs_dict = {aug.name: aug for aug in target_augs}

            different_lengths = (
                len(source_augs_dict) < len(source_augs)
                or len(target_augs_dict) < len(target_augs))
            if different_lengths:
                ia.warn(
                    "Matching mode 'name' with recursive=True was chosen in "
                    "copy_random_state_, but either the source or target "
                    "augmentation sequence contains multiple augmenters with "
                    "the same name."
                )

            for name in target_augs_dict:
                if name in source_augs_dict:
                    if source_augs_dict[name].random_state.is_global_rng():
                        raise Exception(global_rs_exc_msg)
                    # has to be copy(), see above
                    target_augs_dict[name].random_state = \
                        source_augs_dict[name].random_state.copy()
                    if copy_determinism:
                        target_augs_dict[name].deterministic = \
                            source_augs_dict[name].deterministic
                elif not matching_tolerant:
                    raise Exception(
                        "Augmenter name '%s' not found among source "
                        "augmenters." % (name,))
        elif matching == "position":
            if len(source_augs) != len(target_augs) and not matching_tolerant:
                raise Exception(
                    "Source and target augmentation sequences have different "
                    "lengths.")
            for source_aug, target_aug in zip(source_augs, target_augs):
                if source_aug.random_state.is_global_rng():
                    raise Exception(global_rs_exc_msg)
                # has to be copy(), see above
                target_aug.random_state = source_aug.random_state.copy()
                if copy_determinism:
                    target_aug.deterministic = source_aug.deterministic
        else:
            raise Exception(
                "Unknown matching method '%s'. Valid options are 'name' "
                "and 'position'." % (matching,))

        return self

    @abstractmethod
    def get_parameters(self):
        """Get the parameters of this augmenter.

        Returns
        -------
        list
            List of parameters of arbitrary types (usually child class
            of :class:`~imgaug.parameters.StochasticParameter`, but not
            guaranteed to be).

        """
        raise NotImplementedError()

    def get_children_lists(self):
        """Get a list of lists of children of this augmenter.

        For most augmenters, the result will be a single empty list.
        For augmenters with children it will often be a list with one
        sublist containing all children. In some cases the augmenter will
        contain multiple distinct lists of children, e.g. an if-list and an
        else-list. This will lead to a result consisting of a single list
        with multiple sublists, each representing the respective sublist of
        children.

        E.g. for an if/else-augmenter that executes the children ``A1``,
        ``A2`` if a condition is met and otherwise executes the children
        ``B1``, ``B2``, ``B3`` the result will be
        ``[[A1, A2], [B1, B2, B3]]``.

        IMPORTANT: While the topmost list may be newly created, each of the
        sublist must be editable inplace resulting in a changed children list
        of the augmenter. E.g. if an Augmenter
        ``IfElse(condition, [A1, A2], [B1, B2, B3])`` returns
        ``[[A1, A2], [B1, B2, B3]]``
        for a call to
        :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists` and
        ``A2`` is removed inplace from ``[A1, A2]``, then the children lists
        of ``IfElse(...)`` must also change to ``[A1], [B1, B2, B3]``. This
        is used in
        :func:`~imgaug.augmeneters.meta.Augmenter.remove_augmenters_`.

        Returns
        -------
        list of list of imgaug.augmenters.meta.Augmenter
            One or more lists of child augmenter.
            Can also be a single empty list.

        """
        return []

    # TODO why does this exist? it seems to be identical to
    #      get_children_lists() for flat=False, aside from returning list
    #      copies instead of the same instances as used by the augmenters.
    # TODO this can be simplified using imgaug.imgaug.flatten()?
    def get_all_children(self, flat=False):
        """Get all children of this augmenter as a list.

        If the augmenter has no children, the returned list is empty.

        Parameters
        ----------
        flat : bool
            If set to ``True``, the returned list will be flat.

        Returns
        -------
        list of imgaug.augmenters.meta.Augmenter
            The children as a nested or flat list.

        """
        result = []
        for lst in self.get_children_lists():
            for aug in lst:
                result.append(aug)
                children = aug.get_all_children(flat=flat)
                if len(children) > 0:
                    if flat:
                        result.extend(children)
                    else:
                        result.append(children)
        return result

    def find_augmenters(self, func, parents=None, flat=True):
        """Find augmenters that match a condition.

        This function will compare this augmenter and all of its children
        with a condition. The condition is a lambda function.

        Parameters
        ----------
        func : callable
            A function that receives a
            :class:`~imgaug.augmenters.meta.Augmenter` instance and a list of
            parent :class:`~imgaug.augmenters.meta.Augmenter` instances and
            must return ``True``, if that augmenter is valid match or
            ``False`` otherwise.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            List of parent augmenters.
            Intended for nested calls and can usually be left as ``None``.

        flat : bool, optional
            Whether to return the result as a flat list (``True``)
            or a nested list (``False``). In the latter case, the nesting
            matches each augmenters position among the children.

        Returns
        ----------
        list of imgaug.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> aug = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud")
        >>> ])
        >>> print(aug.find_augmenters(lambda a, parents: a.name == "fliplr"))

        Return the first child augmenter (``Fliplr`` instance).

        """
        if parents is None:
            parents = []

        result = []
        if func(self, parents):
            result.append(self)

        subparents = parents + [self]
        for lst in self.get_children_lists():
            for aug in lst:
                found = aug.find_augmenters(func, parents=subparents,
                                            flat=flat)
                if len(found) > 0:
                    if flat:
                        result.extend(found)
                    else:
                        result.append(found)
        return result

    def find_augmenters_by_name(self, name, regex=False, flat=True):
        """Find augmenter(s) by name.

        Parameters
        ----------
        name : str
            Name of the augmenter(s) to search for.

        regex : bool, optional
            Whether `name` parameter is a regular expression.

        flat : bool, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        """
        return self.find_augmenters_by_names([name], regex=regex, flat=flat)

    def find_augmenters_by_names(self, names, regex=False, flat=True):
        """Find augmenter(s) by names.

        Parameters
        ----------
        names : list of str
            Names of the augmenter(s) to search for.

        regex : bool, optional
            Whether `names` is a list of regular expressions.
            If it is, an augmenter is considered a match if *at least* one
            of these expressions is a match.

        flat : boolean, optional
            See :func:`~imgaug.augmenters.meta.Augmenter.find_augmenters`.

        Returns
        -------
        augmenters : list of imgaug.augmenters.meta.Augmenter
            Nested list if `flat` was set to ``False``.
            Flat list if `flat` was set to ``True``.

        """
        if regex:
            def comparer(aug, _parents):
                for pattern in names:
                    if re.match(pattern, aug.name):
                        return True
                return False

            return self.find_augmenters(comparer, flat=flat)
        return self.find_augmenters(
            lambda aug, parents: aug.name in names, flat=flat)

    # TODO remove copy arg
    # TODO allow first arg to be string name, class type or func
    def remove_augmenters(self, func, copy=True, identity_if_topmost=True,
                          noop_if_topmost=None):
        """Remove this augmenter or children that match a condition.

        Parameters
        ----------
        func : callable
            Condition to match per augmenter.
            The function must expect the augmenter itself and a list of parent
            augmenters and returns ``True`` if that augmenter is supposed to
            be removed, or ``False`` otherwise.
            E.g. ``lambda a, parents: a.name == "fliplr" and len(parents) == 1``
            removes an augmenter with name ``fliplr`` if it is the direct child
            of the augmenter upon which ``remove_augmenters()`` was initially
            called.

        copy : bool, optional
            Whether to copy this augmenter and all if its children before
            removing. If ``False``, removal is performed in-place.

        identity_if_topmost : bool, optional
            If ``True`` and the condition (lambda function) leads to the
            removal of the topmost augmenter (the one this function is called
            on initially), then that topmost augmenter will be replaced by an
            instance of :class:`~imgaug.augmenters.meta.Noop` (i.e. an
            augmenter that doesn't change its inputs). If ``False``, ``None``
            will be returned in these cases.
            This can only be ``False`` if copy is set to ``True``.

        noop_if_topmost : bool, optional
            Deprecated since 0.4.0.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter or None
            This augmenter after the removal was performed.
            ``None`` is returned if the condition was matched for the
            topmost augmenter, `copy` was set to ``True`` and `noop_if_topmost`
            was set to ``False``.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>     iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq = seq.remove_augmenters(lambda a, parents: a.name == "fliplr")

        This removes the augmenter ``Fliplr`` from the ``Sequential``
        object's children.

        """
        if noop_if_topmost is not None:
            ia.warn_deprecated("Parameter 'noop_if_topmost' is deprecated. "
                               "Use 'identity_if_topmost' instead.")
            identity_if_topmost = noop_if_topmost

        if func(self, []):
            if not copy:
                raise Exception(
                    "Inplace removal of topmost augmenter requested, "
                    "which is currently not possible. Set 'copy' to True.")

            if identity_if_topmost:
                return Identity()
            return None

        aug = self if not copy else self.deepcopy()
        aug.remove_augmenters_(func, parents=[])
        return aug

    @ia.deprecated("remove_augmenters_")
    def remove_augmenters_inplace(self, func, parents=None):
        """Old name for :func:`~imgaug.meta.Augmenter.remove_augmenters_`.

        Deprecated since 0.4.0.

        """
        self.remove_augmenters_(func=func, parents=parents)

    # TODO allow first arg to be string name, class type or func
    # TODO remove parents arg + add _remove_augmenters_() with parents arg
    def remove_augmenters_(self, func, parents=None):
        """Remove in-place children of this augmenter that match a condition.

        This is functionally identical to
        :func:`~imgaug.augmenters.meta.remove_augmenters` with
        ``copy=False``, except that it does not affect the topmost augmenter
        (the one on which this function is initially called on).

        Added in 0.4.0.

        Parameters
        ----------
        func : callable
            See :func:`~imgaug.augmenters.meta.Augmenter.remove_augmenters`.

        parents : None or list of imgaug.augmenters.meta.Augmenter, optional
            List of parent :class:`~imgaug.augmenters.meta.Augmenter` instances
            that lead to this augmenter. If ``None``, an empty list will be
            used. This parameter can usually be left empty and will be set
            automatically for children.

        Examples
        --------
        >>> import imgaug.augmenters as iaa
        >>> seq = iaa.Sequential([
        >>>     iaa.Fliplr(0.5, name="fliplr"),
        >>>    iaa.Flipud(0.5, name="flipud"),
        >>> ])
        >>> seq.remove_augmenters_(lambda a, parents: a.name == "fliplr")

        This removes the augmenter ``Fliplr`` from the ``Sequential``
        object's children.

        """
        parents = [] if parents is None else parents
        subparents = parents + [self]
        for lst in self.get_children_lists():
            to_remove = []
            for i, aug in enumerate(lst):
                if func(aug, subparents):
                    to_remove.append((i, aug))

            for count_removed, (i, aug) in enumerate(to_remove):
                del lst[i - count_removed]

            for aug in lst:
                aug.remove_augmenters_(func, subparents)

    def copy(self):
        """Create a shallow copy of this Augmenter instance.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Shallow copy of this Augmenter instance.

        """
        return copy_module.copy(self)

    def deepcopy(self):
        """Create a deep copy of this Augmenter instance.

        Returns
        -------
        imgaug.augmenters.meta.Augmenter
            Deep copy of this Augmenter instance.

        """
        # TODO if this augmenter has child augmenters and multiple of them
        #      use the global random state, then after copying, these
        #      augmenters share a single new random state that is a copy of
        #      the global random state (i.e. all use the same *instance*,
        #      not just state). This can lead to confusing bugs.
        # TODO write a custom copying routine?
        return copy_module.deepcopy(self)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        params = self.get_parameters()
        params_str = ", ".join([param.__str__() for param in params])
        return "%s(name=%s, parameters=[%s], deterministic=%s)" % (
            self.__class__.__name__, self.name, params_str, self.deterministic)


class Sequential(Augmenter, list):
    """List augmenter containing child augmenters to apply to inputs.

    This augmenter is simply a list of other augmenters. To augment an image
    or any other data, it iterates over its children and applies each one
    of them independently to the data. (This also means that the second
    applied augmenter will already receive augmented input data and augment
    it further.)

    This augmenter offers the option to apply its children in random order
    using the `random_order` parameter. This should often be activated as
    it greatly increases the space of possible augmentations.

    .. note::

        You are *not* forced to use :class:`~imgaug.augmenters.meta.Sequential`
        in order to use other augmenters. Each augmenter can be used on its
        own, e.g the following defines an augmenter for horizontal flips and
        then augments a single image:

        >>> import numpy as np
        >>> import imgaug.augmenters as iaa
        >>> image = np.zeros((32, 32, 3), dtype=np.uint8)
        >>> aug = iaa.Fliplr(0.5)
        >>> image_aug = aug.augment_image(image)

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.

    random_order : bool, optional
        Whether to apply the child augmenters in random order.
        If ``True``, the order will be randomly sampled once per batch.

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
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Create a :class:`~imgaug.augmenters.meta.Sequential` that always first
    applies a horizontal flip augmenter and then a vertical flip augmenter.
    Each of these two augmenters has a ``50%`` probability of actually
    flipping the image.

    >>> seq = iaa.Sequential([
    >>>     iaa.Fliplr(0.5),
    >>>     iaa.Flipud(0.5)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Create a :class:`~imgaug.augmenters.meta.Sequential` that sometimes first
    applies a horizontal flip augmenter (followed by a vertical flip
    augmenter) and sometimes first a vertical flip augmenter (followed by a
    horizontal flip augmenter). Again, each of them has a ``50%`` probability
    of actually flipping the image.

    """

    def __init__(self, children=None, random_order=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        Augmenter.__init__(
            self,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `Sequential(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become Sequential's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            assert all([isinstance(child, Augmenter) for child in children]), (
                "Expected all children to be augmenters, got types %s." % (
                    ", ".join([str(type(v)) for v in children])))
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, "
                            "got %s." % (type(children),))

        assert ia.is_single_bool(random_order), (
            "Expected random_order to be boolean, got %s." % (
                type(random_order),))
        self.random_order = random_order

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            if self.random_order:
                order = random_state.permutation(len(self))
            else:
                order = sm.xrange(len(self))

            for index in order:
                batch = self[index].augment_batch_(
                    batch,
                    parents=parents + [self],
                    hooks=hooks
                )
        return batch

    def _to_deterministic(self):
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = self.random_state.derive_rng_()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.random_order]

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        imgaug.augmenters.meta.Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self]

    def __str__(self):
        augs_str = ", ".join([aug.__str__() for aug in self])
        pattern = (
            "%s("
            "name=%s, random_order=%s, children=[%s], deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.name, self.random_order, augs_str,
            self.deterministic)


class SomeOf(Augmenter, list):
    """List augmenter that applies only some of its children to inputs.

    This augmenter is similar to :class:`~imgaug.augmenters.meta.Sequential`,
    but may apply only a fixed or random subset of its child augmenters to
    inputs. E.g. the augmenter could be initialized with a list of 20 child
    augmenters and then apply 5 randomly chosen child augmenters to images.

    The subset of augmenters to apply (and their order) is sampled once
    *per image*. If `random_order` is ``True``, the order will be sampled once
    *per batch* (similar to :class:`~imgaug.augmenters.meta.Sequential`).

    This augmenter currently does not support replacing (i.e. picking the same
    child multiple times) due to implementation difficulties in connection
    with deterministic augmenters.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    n : int or tuple of int or list of int or imgaug.parameters.StochasticParameter or None, optional
        Count of augmenters to apply.

            * If ``int``, then exactly `n` of the child augmenters are applied
              to every image.
            * If tuple of two ``int`` s ``(a, b)``, then a random value will
              be uniformly sampled per image from the discrete interval
              ``[a..b]`` and denote the number of child augmenters to pick
              and apply. ``b`` may be set to ``None``, which is then equivalent
              to ``(a..C)`` with ``C`` denoting the number of children that
              the augmenter has.
            * If ``StochasticParameter``, then ``N`` numbers will be sampled
              for ``N`` images. The parameter is expected to be discrete.
            * If ``None``, then the total number of available children will be
              used (i.e. all children will be applied).

    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        The augmenters to apply to images.
        If this is a list of augmenters, it will be converted to a
        :class:`~imgaug.augmenters.meta.Sequential`.

    random_order : boolean, optional
        Whether to apply the child augmenters in random order.
        If ``True``, the order will be randomly sampled once per batch.

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
    >>> imgs = [np.random.rand(10, 10)]
    >>> seq = iaa.SomeOf(1, [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Apply either ``Fliplr`` or ``Flipud`` to images.

    >>> seq = iaa.SomeOf((1, 3), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ])
    >>> imgs_aug = seq.augment_images(imgs)

    Apply one to three of the listed augmenters (``Fliplr``, ``Flipud``,
    ``GaussianBlur``) to images. They are always applied in the
    provided order, i.e. first ``Fliplr``, second ``Flipud``, third
    ``GaussianBlur``.

    >>> seq = iaa.SomeOf((1, None), [
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0),
    >>>     iaa.GaussianBlur(1.0)
    >>> ], random_order=True)
    >>> imgs_aug = seq.augment_images(imgs)

    Apply one to all of the listed augmenters (``Fliplr``, ``Flipud``,
    ``GaussianBlur``) to images. They are applied in random order, i.e.
    sometimes ``GaussianBlur`` first, followed by ``Fliplr``, sometimes
    ``Fliplr`` followed by ``Flipud`` followed by ``Blur`` etc.
    The order is sampled once per batch.

    """

    def __init__(self, n=None, children=None, random_order=False,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        Augmenter.__init__(
            self,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # TODO use handle_children_list() here?
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Augmenter):
            # this must be separate from `list.__init__(self, children)`,
            # otherwise in `SomeOf(OneOf(...))` the OneOf(...) is
            # interpreted as a list and OneOf's children become SomeOf's
            # children
            list.__init__(self, [children])
        elif ia.is_iterable(children):
            assert all([isinstance(child, Augmenter) for child in children]), (
                "Expected all children to be augmenters, got types %s." % (
                    ", ".join([str(type(v)) for v in children])))
            list.__init__(self, children)
        else:
            raise Exception("Expected None or Augmenter or list of Augmenter, "
                            "got %s." % (type(children),))

        self.n, self.n_mode = self._handle_arg_n(n)

        assert ia.is_single_bool(random_order), (
            "Expected random_order to be boolean, got %s." % (
                type(random_order),))
        self.random_order = random_order

    @classmethod
    def _handle_arg_n(cls, n):
        if ia.is_single_number(n):
            n = int(n)
            n_mode = "deterministic"
        elif n is None:
            n = None
            n_mode = "None"
        elif ia.is_iterable(n):
            assert len(n) == 2, (
                "Expected iterable 'n' to contain exactly two values, "
                "got %d." % (len(n),))
            if ia.is_single_number(n[0]) and n[1] is None:
                n = (int(n[0]), None)
                n_mode = "(int,None)"
            elif ia.is_single_number(n[0]) and ia.is_single_number(n[1]):
                n = iap.DiscreteUniform(int(n[0]), int(n[1]))
                n_mode = "stochastic"
            else:
                raise Exception("Expected tuple of (int, None) or (int, int), "
                                "got %s" % ([type(el) for el in n],))
        elif isinstance(n, iap.StochasticParameter):
            n_mode = "stochastic"
        else:
            raise Exception("Expected int, (int, None), (int, int) or "
                            "StochasticParameter, got %s" % (type(n),))
        return n, n_mode

    def _get_n(self, nb_images, random_state):
        if self.n_mode == "deterministic":
            return [self.n] * nb_images
        if self.n_mode == "None":
            return [len(self)] * nb_images
        if self.n_mode == "(int,None)":
            param = iap.DiscreteUniform(self.n[0], len(self))
            return param.draw_samples((nb_images,), random_state=random_state)
        if self.n_mode == "stochastic":
            return self.n.draw_samples((nb_images,), random_state=random_state)
        raise Exception("Invalid n_mode: %s" % (self.n_mode,))

    def _get_augmenter_order(self, random_state):
        if not self.random_order:
            augmenter_order = np.arange(len(self))
        else:
            augmenter_order = random_state.permutation(len(self))
        return augmenter_order

    def _get_augmenter_active(self, nb_rows, random_state):
        # pylint: disable=invalid-name
        nn = self._get_n(nb_rows, random_state)
        nn = [min(n, len(self)) for n in nn]
        augmenter_active = np.zeros((nb_rows, len(self)), dtype=np.bool)
        for row_idx, n_true in enumerate(nn):
            if n_true > 0:
                augmenter_active[row_idx, 0:n_true] = 1
        for row in augmenter_active:
            random_state.shuffle(row)
        return augmenter_active

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            # This must happen before creating the augmenter_active array,
            # otherwise in case of determinism the number of augmented images
            # would change the random_state's state, resulting in the order
            # being dependent on the number of augmented images (and not be
            # constant). By doing this first, the random state is always the
            # same (when determinism is active), so the order is always the
            # same.
            augmenter_order = self._get_augmenter_order(random_state)

            # create an array of active augmenters per image
            # e.g.
            #  [[0, 0, 1],
            #   [1, 0, 1],
            #   [1, 0, 0]]
            # would signal, that augmenter 3 is active for the first image,
            # augmenter 1 and 3 for the 2nd image and augmenter 1 for the 3rd.
            augmenter_active = self._get_augmenter_active(batch.nb_rows,
                                                          random_state)

            for augmenter_index in augmenter_order:
                active = augmenter_active[:, augmenter_index].nonzero()[0]

                if len(active) > 0:
                    batch_sub = batch.subselect_rows_by_indices(active)
                    batch_sub = self[augmenter_index].augment_batch_(
                        batch_sub,
                        parents=parents + [self],
                        hooks=hooks
                    )
                    batch = batch.invert_subselect_rows_by_indices_(active,
                                                                    batch_sub)

            return batch

    def _to_deterministic(self):
        augs = [aug.to_deterministic() for aug in self]
        seq = self.copy()
        seq[:] = augs
        seq.random_state = self.random_state.derive_rng_()
        seq.deterministic = True
        return seq

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.n]

    def add(self, augmenter):
        """Add an augmenter to the list of child augmenters.

        Parameters
        ----------
        augmenter : imgaug.augmenters.meta.Augmenter
            The augmenter to add.

        """
        self.append(augmenter)

    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self]

    def __str__(self):
        augs_str = ", ".join([aug.__str__() for aug in self])
        pattern = (
            "%s("
            "name=%s, n=%s, random_order=%s, augmenters=[%s], deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.name, str(self.n),
            str(self.random_order), augs_str, self.deterministic)


class OneOf(SomeOf):
    """Augmenter that always executes exactly one of its children.

    **Supported dtypes**:

    See :class:`imgaug.augmenters.meta.SomeOf`.

    Parameters
    ----------
    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter
        The choices of augmenters to apply.

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
    >>> images = [np.ones((10, 10), dtype=np.uint8)]  # dummy example images
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Flipud(1.0)
    >>> ])
    >>> images_aug = seq.augment_images(images)

    Flip each image either horizontally or vertically.

    >>> images = [np.ones((10, 10), dtype=np.uint8)]  # dummy example images
    >>> seq = iaa.OneOf([
    >>>     iaa.Fliplr(1.0),
    >>>     iaa.Sequential([
    >>>         iaa.GaussianBlur(1.0),
    >>>         iaa.Dropout(0.05),
    >>>         iaa.AdditiveGaussianNoise(0.1*255)
    >>>     ]),
    >>>     iaa.Noop()
    >>> ])
    >>> images_aug = seq.augment_images(images)

    Either flip each image horizontally, or add blur+dropout+noise or do
    nothing.

    """

    def __init__(self, children,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(OneOf, self).__init__(
            n=1,
            children=children,
            random_order=False,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Sometimes(Augmenter):
    """Apply child augmenter(s) with a probability of `p`.

    Let ``C`` be one or more child augmenters given to
    :class:`~imgaug.augmenters.meta.Sometimes`.
    Let ``p`` be the fraction of images (or other data) to augment.
    Let ``I`` be the input images (or other data).
    Let ``N`` be the number of input images (or other entities).
    Then (on average) ``p*N`` images of ``I`` will be augmented using ``C``.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter, optional
        Sets the probability with which the given augmenters will be applied to
        input images/data. E.g. a value of ``0.5`` will result in ``50%`` of
        all input images (or other augmentables) being augmented.

    then_list : None or imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to `p%` percent of all images.
        If this is a list of augmenters, it will be converted to a
        :class:`~imgaug.augmenters.meta.Sequential`.

    else_list : None or imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter, optional
        Augmenter(s) to apply to ``(1-p)`` percent of all images.
        These augmenters will be applied only when the ones in `then_list`
        are *not* applied (either-or-relationship).
        If this is a list of augmenters, it will be converted to a
        :class:`~imgaug.augmenters.meta.Sequential`.

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
    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3))

    Apply ``GaussianBlur`` to ``50%`` of all input images.

    >>> aug = iaa.Sometimes(0.5, iaa.GaussianBlur(0.3), iaa.Fliplr(1.0))

    Apply ``GaussianBlur`` to ``50%`` of all input images. Apply ``Fliplr``
    to the other ``50%`` of all input images.

    """

    def __init__(self, p=0.5, then_list=None, else_list=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Sometimes, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.p = iap.handle_probability_param(p, "p")

        self.then_list = handle_children_list(then_list, self.name, "then",
                                              default=None)
        self.else_list = handle_children_list(else_list, self.name, "else",
                                              default=None)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        with batch.propagation_hooks_ctx(self, hooks, parents):
            samples = self.p.draw_samples((batch.nb_rows,),
                                          random_state=random_state)

            # create lists/arrays of images for if and else lists (one for each)
            # note that np.where returns tuple(array([0, 5, 9, ...])) or
            # tuple(array([]))
            indices_then_list = np.where(samples == 1)[0]
            indices_else_list = np.where(samples == 0)[0]

            indice_lists = [indices_then_list, indices_else_list]
            augmenter_lists = [self.then_list, self.else_list]

            # For then_list: collect augmentables to be processed by then_list
            # augmenters, apply them to the list, then map back to the output
            # list. Analogous for else_list.
            # TODO maybe this would be easier if augment_*() accepted a list
            #      that can contain Nones
            for indices, augmenters in zip(indice_lists, augmenter_lists):
                if augmenters is not None and len(augmenters) > 0:
                    batch_sub = batch.subselect_rows_by_indices(indices)
                    batch_sub = augmenters.augment_batch_(
                        batch_sub,
                        parents=parents + [self],
                        hooks=hooks
                    )
                    batch = batch.invert_subselect_rows_by_indices_(indices,
                                                                    batch_sub)

            return batch

    def _to_deterministic(self):
        aug = self.copy()
        aug.then_list = (aug.then_list.to_deterministic()
                         if aug.then_list is not None
                         else aug.then_list)
        aug.else_list = (aug.else_list.to_deterministic()
                         if aug.else_list is not None
                         else aug.else_list)
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p]

    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        result = []
        if self.then_list is not None:
            result.append(self.then_list)
        if self.else_list is not None:
            result.append(self.else_list)
        return result

    def __str__(self):
        pattern = (
            "%s("
            "p=%s, name=%s, then_list=%s, else_list=%s, deterministic=%s"
            ")")
        return pattern % (
            self.__class__.__name__, self.p, self.name, self.then_list,
            self.else_list, self.deterministic)


class WithChannels(Augmenter):
    """Apply child augmenters to specific channels.

    Let ``C`` be one or more child augmenters given to this augmenter.
    Let ``H`` be a list of channels.
    Let ``I`` be the input images.
    Then this augmenter will pick the channels ``H`` from each image
    in ``I`` (resulting in new images) and apply ``C`` to them.
    The result of the augmentation will be merged back into the original
    images.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    channels : None or int or list of int, optional
        Sets the channels to be extracted from each image.
        If ``None``, all channels will be used. Note that this is not
        stochastic - the extracted channels are always the same ones.

    children : imgaug.augmenters.meta.Augmenter or list of imgaug.augmenters.meta.Augmenter or None, optional
        One or more augmenters to apply to images, after the channels
        are extracted.

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
    >>> aug = iaa.WithChannels([0], iaa.Add(10))

    Assuming input images are RGB, then this augmenter will add ``10`` only to
    the first channel, i.e. it will make images appear more red.

    """

    def __init__(self, channels=None, children=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(WithChannels, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        # TODO change this to a stochastic parameter
        if channels is None:
            self.channels = None
        elif ia.is_single_integer(channels):
            self.channels = [channels]
        elif ia.is_iterable(channels):
            only_ints = all([
                ia.is_single_integer(channel) for channel in channels])
            assert only_ints, (
                "Expected integers as channels, got %s." % (
                    [type(channel) for channel in channels],))
            self.channels = channels
        else:
            raise Exception("Expected None, int or list of ints as channels, "
                            "got %s." % (type(channels),))

        self.children = handle_children_list(children, self.name, "then")

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if self.channels is not None and len(self.channels) == 0:
            return batch

        with batch.propagation_hooks_ctx(self, hooks, parents):
            batch_cp = batch.deepcopy()

            if batch.images is not None:
                batch.images = self._reduce_images_to_channels(batch.images)

            # Note that we augment here all data, including non-image data
            # for which less than 50% of the corresponding image channels
            # were augmented. This is because (a) the system does not yet
            # understand None as cell values and (b) decreasing the length
            # of columns leads to potential RNG misalignments.
            # We replace non-image data that was not supposed to be augmented
            # further below.
            batch = self.children.augment_batch_(
                batch, parents=parents + [self], hooks=hooks)

            # If the shapes changed we cannot insert the augmented channels
            # into the existing ones as the shapes of the non-augmented
            # channels are still the same.
            if batch.images is not None:
                self._assert_lengths_not_changed(batch.images, batch_cp.images)
                self._assert_shapes_not_changed(batch.images, batch_cp.images)
                self._assert_dtypes_not_changed(batch.images, batch_cp.images)

                batch.images = self._recover_images_array(batch.images,
                                                          batch_cp.images)

            for column in batch.columns:
                if column.name != "images":
                    value_old = getattr(batch_cp, column.attr_name)
                    value = self._replace_unaugmented_cells(column.value,
                                                            value_old)
                    setattr(batch, column.attr_name, value)

            if batch.images is not None:
                batch.images = self._invert_reduce_images_to_channels(
                    batch.images, batch_cp.images)

        return batch

    # Added in 0.4.0.
    @classmethod
    def _assert_lengths_not_changed(cls, images_aug, images):
        assert len(images_aug) == len(images), (
            "Expected that number of images does not change during "
            "augmentation, but got %d vs. originally %d images." % (
                len(images_aug), len(images)))

    # Added in 0.4.0.
    @classmethod
    def _assert_shapes_not_changed(cls, images_aug, images):
        if ia.is_np_array(images_aug) and ia.is_np_array(images):
            shapes_same = (images_aug.shape[1:3] == images.shape[1:3])
        else:
            shapes_same = all(
                [image_aug.shape[0:2] == image.shape[0:2]
                 for image_aug, image
                 in zip(images_aug, images)])
        assert shapes_same, (
            "Heights/widths of images changed in WithChannels from "
            "%s to %s, but expected to be the same." % (
                str([image.shape[0:2] for image in images]),
                str([image_aug.shape[0:2] for image_aug in images_aug]),
            ))

    # Added in 0.4.0.
    @classmethod
    def _assert_dtypes_not_changed(cls, images_aug, images):
        if ia.is_np_array(images_aug) and ia.is_np_array(images):
            dtypes_same = (images_aug.dtype.name == images.dtype.name)
        else:
            dtypes_same = all(
                [image_aug.dtype.name == image.dtype.name
                 for image_aug, image
                 in zip(images_aug, images)])

        assert dtypes_same, (
            "dtypes of images changed in WithChannels from "
            "%s to %s, but expected to be the same." % (
                str([image.dtype.name for image in images]),
                str([image_aug.dtype.name for image_aug in images_aug]),
            ))

    # Added in 0.4.0.
    @classmethod
    def _recover_images_array(cls, images_aug, images):
        if ia.is_np_array(images):
            return np.array(images_aug)
        return images_aug

    # Added in 0.4.0.
    def _reduce_images_to_channels(self, images):
        if self.channels is None:
            return images
        if ia.is_np_array(images):
            return images[..., self.channels]
        return [image[..., self.channels] for image in images]

    # Added in 0.4.0.
    def _invert_reduce_images_to_channels(self, images_aug, images):
        if self.channels is None:
            return images_aug

        for image, image_aug in zip(images, images_aug):
            image[..., self.channels] = image_aug
        return images

    # Added in 0.4.0.
    def _replace_unaugmented_cells(self, augmentables_aug, augmentables):
        if self.channels is None:
            return augmentables_aug

        nb_channels_to_aug = len(self.channels)
        nb_channels_lst = [augm.shape[2] if len(augm.shape) > 2 else 1
                           for augm in augmentables]

        # We use the augmented form of a non-image if at least 50% of the
        # corresponding image's channels were augmented. Otherwise we use
        # the unaugmented form.
        fraction_augmented_lst = [nb_channels_to_aug/nb_channels
                                  for nb_channels in nb_channels_lst]
        result = [
            (augmentable_aug if fraction_augmented >= 0.5 else augmentable)
            for augmentable_aug, augmentable, fraction_augmented
            in zip(augmentables_aug, augmentables, fraction_augmented_lst)]
        return result

    def _to_deterministic(self):
        aug = self.copy()
        aug.children = aug.children.to_deterministic()
        aug.deterministic = True
        aug.random_state = self.random_state.derive_rng_()
        return aug

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.channels]

    def get_children_lists(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_children_lists`."""
        return [self.children]

    def __str__(self):
        pattern = (
            "%s("
            "channels=%s, name=%s, children=%s, deterministic=%s"
            ")")
        return pattern % (self.__class__.__name__, self.channels, self.name,
                          self.children, self.deterministic)


class Identity(Augmenter):
    """Augmenter that does not change the input data.

    This augmenter is useful e.g. during validation/testing as it allows
    to re-use the training code without actually performing any augmentation.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
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
    >>> aug = iaa.Identity()

    Create an augmenter that does not change inputs.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Identity, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        return batch

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class Noop(Identity):
    """Alias for augmenter :class:`Identity`.

    It is recommended to now use :class:`Identity`. :class:`Noop` might be
    deprecated in the future.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.meta.Identity`.

    Parameters
    ----------
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
    >>> aug = iaa.Noop()

    Create an augmenter that does not change inputs.

    """

    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Noop, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class Lambda(Augmenter):
    """Augmenter that calls a lambda function for each input batch.

    This is useful to add missing functions to a list of augmenters.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    func_images : None or callable, optional
        The function to call for each batch of images.
        It must follow the form::

            function(images, random_state, parents, hooks)

        and return the changed images (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_images`.
        If this is ``None`` instead of a function, the images will not be
        altered.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form::

            function(heatmaps, random_state, parents, hooks)

        and return the changed heatmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_heatmaps`.
        If this is ``None`` instead of a function, the heatmaps will not be
        altered.

    func_segmentation_maps : None or callable, optional
        The function to call for each batch of segmentation maps.
        It must follow the form::

            function(segmaps, random_state, parents, hooks)

        and return the changed segmaps (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_segmentation_maps`.
        If this is ``None`` instead of a function, the segmentatio maps will
        not be altered.

    func_keypoints : None or callable, optional
        The function to call for each batch of keypoints.
        It must follow the form::

            function(keypoints_on_images, random_state, parents, hooks)

        and return the changed keypoints (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_keypoints`.
        If this is ``None`` instead of a function, the keypoints will not be
        altered.

    func_bounding_boxes : "keypoints" or None or callable, optional
        The function to call for each batch of bounding boxes.
        It must follow the form::

            function(bounding_boxes_on_images, random_state, parents, hooks)

        and return the changed bounding boxes (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_bounding_boxes`.
        If this is ``None`` instead of a function, the bounding boxes will not
        be altered.
        If this is the string ``"keypoints"`` instead of a function, the
        bounding boxes will automatically be augmented by transforming their
        corner vertices to keypoints and calling `func_keypoints`.

        Added in 0.4.0.

    func_polygons : "keypoints" or None or callable, optional
        The function to call for each batch of polygons.
        It must follow the form::

            function(polygons_on_images, random_state, parents, hooks)

        and return the changed polygons (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_polygons`.
        If this is ``None`` instead of a function, the polygons will not
        be altered.
        If this is the string ``"keypoints"`` instead of a function, the
        polygons will automatically be augmented by transforming their
        corner vertices to keypoints and calling `func_keypoints`.

    func_line_strings : "keypoints" or None or callable, optional
        The function to call for each batch of line strings.
        It must follow the form::

            function(line_strings_on_images, random_state, parents, hooks)

        and return the changed line strings (may be transformed in-place).
        This is essentially the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_line_strings`.
        If this is ``None`` instead of a function, the line strings will not
        be altered.
        If this is the string ``"keypoints"`` instead of a function, the
        line strings will automatically be augmented by transforming their
        corner vertices to keypoints and calling `func_keypoints`.

        Added in 0.4.0.

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
    >>>
    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images
    >>> )

    Replace every second row in input images with black pixels. Leave
    other data (e.g. heatmaps, keypoints) unchanged.

    >>> def func_images(images, random_state, parents, hooks):
    >>>     images[:, ::2, :, :] = 0
    >>>     return images
    >>>
    >>> def func_heatmaps(heatmaps, random_state, parents, hooks):
    >>>     for heatmaps_i in heatmaps:
    >>>         heatmaps.arr_0to1[::2, :, :] = 0
    >>>     return heatmaps
    >>>
    >>> def func_keypoints(keypoints_on_images, random_state, parents, hooks):
    >>>     return keypoints_on_images
    >>>
    >>> aug = iaa.Lambda(
    >>>     func_images=func_images,
    >>>     func_heatmaps=func_heatmaps,
    >>>     func_keypoints=func_keypoints
    >>> )

    Replace every second row in images with black pixels, set every second
    row in heatmaps to zero and leave other data (e.g. keypoints)
    unchanged.

    """

    def __init__(self, func_images=None, func_heatmaps=None,
                 func_segmentation_maps=None, func_keypoints=None,
                 func_bounding_boxes="keypoints", func_polygons="keypoints",
                 func_line_strings="keypoints",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Lambda, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.func_images = func_images
        self.func_heatmaps = func_heatmaps
        self.func_segmentation_maps = func_segmentation_maps
        self.func_keypoints = func_keypoints
        self.func_bounding_boxes = func_bounding_boxes
        self.func_polygons = func_polygons
        self.func_line_strings = func_line_strings

    def _augment_images(self, images, random_state, parents, hooks):
        if self.func_images is not None:
            return self.func_images(images, random_state, parents, hooks)
        return images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        if self.func_heatmaps is not None:
            result = self.func_heatmaps(heatmaps, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for heatmaps to return list of "
                "imgaug.HeatmapsOnImage instances, got %s." % (
                    type(result),))
            only_heatmaps = all([
                isinstance(el, ia.HeatmapsOnImage) for el in result])
            assert only_heatmaps, (
                "Expected callback function for heatmaps to return list of "
                "imgaug.HeatmapsOnImage instances, got %s." % (
                    [type(el) for el in result],))
            return result
        return heatmaps

    def _augment_segmentation_maps(self, segmaps, random_state, parents, hooks):
        if self.func_segmentation_maps is not None:
            result = self.func_segmentation_maps(segmaps, random_state,
                                                 parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for segmentation maps to return "
                "list of imgaug.SegmentationMapsOnImage() instances, "
                "got %s." % (type(result),))
            only_segmaps = all([
                isinstance(el, ia.SegmentationMapsOnImage) for el in result])
            assert only_segmaps, (
                "Expected callback function for segmentation maps to return "
                "list of imgaug.SegmentationMapsOnImage() instances, "
                "got %s." % ([type(el) for el in result],))
            return result
        return segmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        if self.func_keypoints is not None:
            result = self.func_keypoints(keypoints_on_images, random_state,
                                         parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for keypoints to return list of "
                "imgaug.augmentables.kps.KeypointsOnImage instances, "
                "got %s." % (type(result),))
            only_keypoints = all([
                isinstance(el, ia.KeypointsOnImage) for el in result])
            assert only_keypoints, (
                "Expected callback function for keypoints to return list of "
                "imgaug.augmentables.kps.KeypointsOnImage instances, "
                "got %s." % ([type(el) for el in result],))
            return result
        return keypoints_on_images

    def _augment_polygons(self, polygons_on_images, random_state, parents,
                          hooks):
        from imgaug.augmentables.polys import _ConcavePolygonRecoverer

        if self.func_polygons == "keypoints":
            return self._augment_polygons_as_keypoints(
                polygons_on_images, random_state, parents, hooks,
                recoverer=_ConcavePolygonRecoverer())
        if self.func_polygons is not None:
            result = self.func_polygons(polygons_on_images, random_state,
                                        parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for polygons to return list of "
                "imgaug.augmentables.polys.PolygonsOnImage instances, "
                "got %s." % (type(result),))
            only_polygons = all([
                isinstance(el, ia.PolygonsOnImage) for el in result])
            assert only_polygons, (
                "Expected callback function for polygons to return list of "
                "imgaug.augmentables.polys.PolygonsOnImage instances, "
                "got %s." % ([type(el) for el in result],))
            return result
        return polygons_on_images

    # Added in 0.4.0.
    def _augment_line_strings(self, line_strings_on_images, random_state,
                              parents, hooks):
        if self.func_line_strings == "keypoints":
            return self._augment_line_strings_as_keypoints(
                line_strings_on_images, random_state, parents, hooks)
        if self.func_line_strings is not None:
            result = self.func_line_strings(line_strings_on_images,
                                            random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for line strings to return list of "
                "imgaug.augmentables.lines.LineStringsOnImage instances, "
                "got %s." % (type(result),))
            only_ls = all([
                isinstance(el, ia.LineStringsOnImage) for el in result])
            assert only_ls, (
                "Expected callback function for line strings to return list of "
                "imgaug.augmentables.lines.LineStringsOnImages instances, "
                "got %s." % ([type(el) for el in result],))
            return result
        return line_strings_on_images

    # Added in 0.4.0.
    def _augment_bounding_boxes(self, bounding_boxes_on_images, random_state,
                                parents, hooks):
        if self.func_bounding_boxes == "keypoints":
            return self._augment_bounding_boxes_as_keypoints(
                bounding_boxes_on_images, random_state, parents, hooks)
        if self.func_bounding_boxes is not None:
            result = self.func_bounding_boxes(
                bounding_boxes_on_images, random_state, parents, hooks)
            assert ia.is_iterable(result), (
                "Expected callback function for bounding boxes to return list "
                "of imgaug.augmentables.bbs.BoundingBoxesOnImage instances, "
                "got %s." % (type(result),))
            only_bbs = all([
                isinstance(el, ia.BoundingBoxesOnImage) for el in result])
            assert only_bbs, (
                "Expected callback function for polygons to return list of "
                "imgaug.augmentables.polys.PolygonsOnImage instances, "
                "got %s." % ([type(el) for el in result],))

            for bboi in bounding_boxes_on_images:
                for bb in bboi.bounding_boxes:
                    if bb.x1 > bb.x2:
                        bb.x1, bb.x2 = bb.x2, bb.x1
                    if bb.y1 > bb.y2:
                        bb.y1, bb.y2 = bb.y2, bb.y1

            return result
        return bounding_boxes_on_images

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class AssertLambda(Lambda):
    """Assert conditions based on lambda-function to be the case for input data.

    This augmenter applies a lambda function to each image or other input.
    The lambda function must return ``True`` or ``False``. If ``False`` is
    returned, an assertion error is produced.

    This is useful to ensure that generic assumption about the input data
    are actually the case and error out early otherwise.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    func_images : None or callable, optional
        The function to call for each batch of images.
        It must follow the form::

            function(images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_images`.

    func_heatmaps : None or callable, optional
        The function to call for each batch of heatmaps.
        It must follow the form::

            function(heatmaps, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_heatmaps`.

    func_segmentation_maps : None or callable, optional
        The function to call for each batch of segmentation maps.
        It must follow the form::

            function(segmaps, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_segmentation_maps`.

    func_keypoints : None or callable, optional
        The function to call for each batch of keypoints.
        It must follow the form::

            function(keypoints_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_keypoints`.

    func_bounding_boxes : None or callable, optional
        The function to call for each batch of bounding boxes.
        It must follow the form::

            function(bounding_boxes_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_bounding_boxes`.

        Added in 0.4.0.

    func_polygons : None or callable, optional
        The function to call for each batch of polygons.
        It must follow the form::

            function(polygons_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_polygons`.

    func_line_strings : None or callable, optional
        The function to call for each batch of line strings.
        It must follow the form::

            function(line_strings_on_images, random_state, parents, hooks)

        and return either ``True`` (valid input) or ``False`` (invalid input).
        It essentially re-uses the interface of
        :func:`~imgaug.augmenters.meta.Augmenter._augment_line_strings`.

        Added in 0.4.0.

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

    """

    def __init__(self, func_images=None, func_heatmaps=None,
                 func_segmentation_maps=None, func_keypoints=None,
                 func_bounding_boxes=None, func_polygons=None,
                 func_line_strings=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        def _default(var, augmentable_name):
            return (
                _AssertLambdaCallback(var, augmentable_name=augmentable_name)
                if var is not None
                else None
            )

        super(AssertLambda, self).__init__(
            func_images=_default(func_images, "images"),
            func_heatmaps=_default(func_heatmaps, "heatmaps"),
            func_segmentation_maps=_default(func_segmentation_maps,
                                            "segmentation_maps"),
            func_keypoints=_default(func_keypoints, "keypoints"),
            func_bounding_boxes=_default(func_bounding_boxes, "bounding_boxes"),
            func_polygons=_default(func_polygons, "polygons"),
            func_line_strings=_default(func_line_strings, "line_strings"),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


# Added in 0.4.0.
class _AssertLambdaCallback(object):
    # Added in 0.4.0.
    def __init__(self, func, augmentable_name):
        self.func = func
        self.augmentable_name = augmentable_name

    # Added in 0.4.0.
    def __call__(self, augmentables, random_state, parents, hooks):
        assert self.func(augmentables, random_state, parents, hooks), (
            "Input %s did not fulfill user-defined assertion in "
            "AssertLambda." % (self.augmentable_name,))
        return augmentables


# TODO add tests for segmaps
# TODO This evaluates .shape for kps/polys, but the array shape for
#      heatmaps/segmaps. Not very consistent.
class AssertShape(Lambda):
    """Assert that inputs have a specified shape.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    shape : tuple
        The expected shape, given as a ``tuple``. The number of entries in
        the ``tuple`` must match the number of dimensions, i.e. it must
        contain four entries for ``(N, H, W, C)``. If only a single entity
        is augmented, e.g. via
        :func:`~imgaug.augmenters.meta.Augmenter.augment_image`, then ``N`` is
        ``1`` in the input to this augmenter. Images that don't have
        a channel axis will automatically have one assigned, i.e. ``C`` is
        at least ``1``.
        For each component of the ``tuple`` one of the following datatypes
        may be used:

            * If a component is ``None``, any value for that dimensions is
              accepted.
            * If a component is ``int``, exactly that value (and no other one)
              will be accepted for that dimension.
            * If a component is a ``tuple`` of two ``int`` s with values ``a``
              and ``b``, only a value within the interval ``[a, b)`` will be
              accepted for that dimension.
            * If an entry is a ``list`` of ``int`` s, only a value from that
              ``list`` will be accepted for that dimension.

    check_images : bool, optional
        Whether to validate input images via the given shape.

    check_heatmaps : bool, optional
        Whether to validate input heatmaps via the given shape.
        The number of heatmaps will be verified as ``N``. For each
        :class:`~imgaug.augmentables.heatmaps.HeatmapsOnImage` instance
        its array's height and width will be verified as ``H`` and ``W``,
        but not the channel count.

    check_segmentation_maps : bool, optional
        Whether to validate input segmentation maps via the given shape.
        The number of segmentation maps will be verified as ``N``. For each
        :class:`~imgaug.augmentables.segmaps.SegmentationMapOnImage` instance
        its array's height and width will be verified as ``H`` and ``W``,
        but not the channel count.

    check_keypoints : bool, optional
        Whether to validate input keypoints via the given shape.
        This will check (a) the number of keypoints and (b) for each
        :class:`~imgaug.augmentables.kps.KeypointsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

    check_bounding_boxes : bool, optional
        Whether to validate input bounding boxes via the given shape.
        This will check (a) the number of bounding boxes and (b) for each
        :class:`~imgaug.augmentables.bbs.BoundingBoxesOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

        Added in 0.4.0.

    check_polygons : bool, optional
        Whether to validate input polygons via the given shape.
        This will check (a) the number of polygons and (b) for each
        :class:`~imgaug.augmentables.polys.PolygonsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

    check_line_strings : bool, optional
        Whether to validate input line strings via the given shape.
        This will check (a) the number of line strings and (b) for each
        :class:`~imgaug.augmentables.lines.LineStringsOnImage` instance the
        ``.shape`` attribute, i.e. the shape of the corresponding image.

        Added in 0.4.0.

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
    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, 32, 32, 3)),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    Verify first for each image batch if it contains a variable number of
    ``32x32`` images with ``3`` channels each. Only if that check succeeds, the
    horizontal flip will be executed. Otherwise an assertion error will be
    raised.

    >>> seq = iaa.Sequential([
    >>>     iaa.AssertShape((None, (32, 64), 32, [1, 3])),
    >>>     iaa.Fliplr(0.5)
    >>> ])

    Similar to the above example, but now the height may be in the interval
    ``[32, 64)`` and the number of channels may be either ``1`` or ``3``.

    """

    def __init__(self, shape, check_images=True, check_heatmaps=True,
                 check_segmentation_maps=True, check_keypoints=True,
                 check_bounding_boxes=True, check_polygons=True,
                 check_line_strings=True,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        assert len(shape) == 4, (
            "Expected shape to have length 4, got %d with shape: %s." % (
                len(shape), str(shape)))
        self.shape = shape

        def _default(func, do_use):
            return func if do_use else None

        super(AssertShape, self).__init__(
            func_images=_default(_AssertShapeImagesCheck(shape),
                                 check_images),
            func_heatmaps=_default(_AssertShapeHeatmapsCheck(shape),
                                   check_heatmaps),
            func_segmentation_maps=_default(_AssertShapeSegmapCheck(shape),
                                            check_segmentation_maps),
            func_keypoints=_default(_AssertShapeKeypointsCheck(shape),
                                    check_keypoints),
            func_bounding_boxes=_default(_AssertShapeBoundingBoxesCheck(shape),
                                         check_bounding_boxes),
            func_polygons=_default(_AssertShapePolygonsCheck(shape),
                                   check_polygons),
            func_line_strings=_default(_AssertShapeLineStringsCheck(shape),
                                       check_line_strings),
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    @classmethod
    def _compare(cls, observed, expected, dimension, image_index):
        if expected is not None:
            if ia.is_single_integer(expected):
                assert observed == expected, (
                    "Expected dim %d (entry index: %s) to have value %d, "
                    "got %d." % (dimension, image_index, expected,
                                 observed))
            elif isinstance(expected, tuple):
                assert len(expected) == 2, (
                    "Expected tuple argument 'expected' to contain "
                    "exactly 2 entries, got %d." % (len(expected),))
                assert expected[0] <= observed < expected[1], (
                    "Expected dim %d (entry index: %s) to have value in "
                    "interval [%d, %d), got %d." % (
                        dimension, image_index, expected[0], expected[1],
                        observed))
            elif isinstance(expected, list):
                assert any([observed == val for val in expected]), (
                    "Expected dim %d (entry index: %s) to have any value "
                    "of %s, got %d." % (
                        dimension, image_index, str(expected), observed))
            else:
                raise Exception(
                    "Invalid datatype for shape entry %d, expected each "
                    "entry to be an integer, a tuple (with two entries) "
                    "or a list, got %s." % (dimension, type(expected),))

    @classmethod
    def _check_shapes(cls, shapes, shape_target):
        if shape_target[0] is not None:
            cls._compare(len(shapes), shape_target[0], 0, "ALL")

        for augm_idx, shape in enumerate(shapes):
            # note that dim_idx is here per object, dim 0 of shape target
            # denotes "number of all objects" and was checked above
            for dim_idx, expected in enumerate(shape_target[1:]):
                observed = shape[dim_idx]
                cls._compare(observed, expected, dim_idx, augm_idx)


# turning these checks below into classmethods of AssertShape breaks pickling
# in python 2.7
# Added in 0.4.0.
class _AssertShapeImagesCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, images, _random_state, _parents, _hooks):
        # set shape_target so that we check all target dimensions,
        # including C, which isn't checked for the other methods
        AssertShape._check_shapes([obj.shape for obj in images],
                                  self.shape)
        return images


# Added in 0.4.0.
class _AssertShapeHeatmapsCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, heatmaps, _random_state, _parents, _hooks):
        AssertShape._check_shapes([obj.arr_0to1.shape for obj in heatmaps],
                                  self.shape[0:3])
        return heatmaps


# Added in 0.4.0.
class _AssertShapeSegmapCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, segmaps, _random_state, _parents, _hooks):
        AssertShape._check_shapes([obj.arr.shape for obj in segmaps],
                                  self.shape[0:3])
        return segmaps


# Added in 0.4.0.
class _AssertShapeKeypointsCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, keypoints_on_images, _random_state, _parents, _hooks):
        AssertShape._check_shapes([obj.shape for obj in keypoints_on_images],
                                  self.shape[0:3])
        return keypoints_on_images


# Added in 0.4.0.
class _AssertShapeBoundingBoxesCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, bounding_boxes_on_images, _random_state, _parents,
                 _hooks):
        AssertShape._check_shapes([obj.shape for obj
                                   in bounding_boxes_on_images],
                                  self.shape[0:3])
        return bounding_boxes_on_images


# Added in 0.4.0.
class _AssertShapePolygonsCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, polygons_on_images, _random_state, _parents, _hooks):
        AssertShape._check_shapes([obj.shape for obj in polygons_on_images],
                                  self.shape[0:3])
        return polygons_on_images


# Added in 0.4.0.
class _AssertShapeLineStringsCheck(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, line_strings_on_images, _random_state, _parents,
                 _hooks):
        AssertShape._check_shapes([obj.shape for obj
                                   in line_strings_on_images],
                                  self.shape[0:3])
        return line_strings_on_images


class ChannelShuffle(Augmenter):
    """Randomize the order of channels in input images.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: yes; tested
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: yes; tested
        * ``float16``: yes; tested
        * ``float32``: yes; tested
        * ``float64``: yes; tested
        * ``float128``: yes; tested
        * ``bool``: yes; tested

    Parameters
    ----------
    p : float or imgaug.parameters.StochasticParameter, optional
        Probability of shuffling channels in any given image.
        May be a fixed probability as a ``float``, or a
        :class:`~imgaug.parameters.StochasticParameter` that returns ``0`` s
        and ``1`` s.

    channels : None or imgaug.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug.ALL``, then all channels may be
        shuffled. If it is a ``list`` of ``int`` s,
        then only the channels with indices in that list may be shuffled.
        (Values start at ``0``. All channel indices in the list must exist in
        each image.)

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
    >>> aug = iaa.ChannelShuffle(0.35)

    Shuffle all channels of ``35%`` of all images.

    >>> aug = iaa.ChannelShuffle(0.35, channels=[0, 1])

    Shuffle only channels ``0`` and ``1`` of ``35%`` of all images. As the new
    channel orders ``0, 1`` and ``1, 0`` are both valid outcomes of the
    shuffling, it means that for ``0.35 * 0.5 = 0.175`` or ``17.5%`` of all
    images the order of channels ``0`` and ``1`` is inverted.

    """

    def __init__(self, p=1.0, channels=None,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ChannelShuffle, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.p = iap.handle_probability_param(p, "p")
        valid_channels = (
            channels is None
            or channels == ia.ALL
            or (
                isinstance(channels, list)
                and all([ia.is_single_integer(v) for v in channels])
            ))
        assert valid_channels, (
            "Expected None or imgaug.ALL or list of int, got %s." % (
                type(channels),))
        self.channels = channels

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        nb_images = len(images)
        p_samples = self.p.draw_samples((nb_images,),
                                        random_state=random_state)
        rss = random_state.duplicate(nb_images)
        for i, (image, p_i, rs) in enumerate(zip(images, p_samples, rss)):
            if p_i >= 1-1e-4:
                batch.images[i] = shuffle_channels(image, rs, self.channels)
        return batch

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.channels]


def shuffle_channels(image, random_state, channels=None):
    """Randomize the order of (color) channels in an image.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; indirectly tested (1)
        * ``uint32``: yes; indirectly tested (1)
        * ``uint64``: yes; indirectly tested (1)
        * ``int8``: yes; indirectly tested (1)
        * ``int16``: yes; indirectly tested (1)
        * ``int32``: yes; indirectly tested (1)
        * ``int64``: yes; indirectly tested (1)
        * ``float16``: yes; indirectly tested (1)
        * ``float32``: yes; indirectly tested (1)
        * ``float64``: yes; indirectly tested (1)
        * ``float128``: yes; indirectly tested (1)
        * ``bool``: yes; indirectly tested (1)

        - (1) Indirectly tested via :class:`ChannelShuffle`.

    Parameters
    ----------
    image : (H,W,[C]) ndarray
        Image of any dtype for which to shuffle the channels.

    random_state : imgaug.random.RNG
        The random state to use for this shuffling operation.

    channels : None or imgaug.ALL or list of int, optional
        Which channels are allowed to be shuffled with each other.
        If this is ``None`` or ``imgaug.ALL``, then all channels may be
        shuffled. If it is a ``list`` of ``int`` s,
        then only the channels with indices in that list may be shuffled.
        (Values start at ``0``. All channel indices in the list must exist in
        the image.)

    Returns
    -------
    ndarray
        The input image with shuffled channels.

    """
    if image.ndim < 3 or image.shape[2] == 1:
        return image
    nb_channels = image.shape[2]
    all_channels = np.arange(nb_channels)
    is_all_channels = (
        channels is None
        or channels == ia.ALL
        or len(set(all_channels).difference(set(channels))) == 0
    )
    if is_all_channels:
        # note that if this is the case, then 'channels' may be None or
        # imgaug.ALL, so don't simply move the assignment outside of the
        # if/else
        channels_perm = random_state.permutation(all_channels)
        return image[..., channels_perm]

    channels_perm = random_state.permutation(channels)
    channels_perm_full = all_channels
    for channel_source, channel_target in zip(channels, channels_perm):
        channels_perm_full[channel_source] = channel_target
    return image[..., channels_perm_full]


class RemoveCBAsByOutOfImageFraction(Augmenter):
    """Remove coordinate-based augmentables exceeding an out of image fraction.

    This augmenter inspects all coordinate-based augmentables (e.g.
    bounding boxes, line strings) within a given batch and removes any such
    augmentable which's out of image fraction is exactly a given value or
    greater than that. The out of image fraction denotes the fraction of the
    augmentable's area that is outside of the image, e.g. for a bounding box
    that has half of its area outside of the image it would be ``0.5``.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
    fraction : number
        Remove any augmentable for which ``fraction_{actual} >= fraction``,
        where ``fraction_{actual}`` denotes the estimated out of image
        fraction.

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
    >>> aug = iaa.Sequential([
    >>>     iaa.Affine(translate_px={"x": (-100, 100)}),
    >>>     iaa.RemoveCBAsByOutOfImageFraction(0.5)
    >>> ])

    Translate all inputs by ``-100`` to ``100`` pixels on the x-axis, then
    remove any coordinate-based augmentable (e.g. bounding boxes) which has
    at least ``50%`` of its area outside of the image plane.

    >>> import imgaug as ia
    >>> import imgaug.augmenters as iaa
    >>> image = ia.quokka_square((100, 100))
    >>> bb = ia.BoundingBox(x1=50-25, y1=0, x2=50+25, y2=100)
    >>> bbsoi = ia.BoundingBoxesOnImage([bb], shape=image.shape)
    >>> aug_without = iaa.Affine(translate_px={"x": 51})
    >>> aug_with = iaa.Sequential([
    >>>     iaa.Affine(translate_px={"x": 51}),
    >>>     iaa.RemoveCBAsByOutOfImageFraction(0.5)
    >>> ])
    >>>
    >>> image_without, bbsoi_without = aug_without(
    >>>     image=image, bounding_boxes=bbsoi)
    >>> image_with, bbsoi_with = aug_with(
    >>>     image=image, bounding_boxes=bbsoi)
    >>>
    >>> assert len(bbsoi_without.bounding_boxes) == 1
    >>> assert len(bbsoi_with.bounding_boxes) == 0

    Create a bounding box on an example image, then translate the image so that
    ``50%`` of the bounding box's area is outside of the image and compare
    the effects and using ``RemoveCBAsByOutOfImageFraction`` with not using it.

    """

    # Added in 0.4.0.
    def __init__(self, fraction,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(RemoveCBAsByOutOfImageFraction, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.fraction = fraction

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        for column in batch.columns:
            if column.name in ["keypoints", "bounding_boxes", "polygons",
                               "line_strings"]:
                for i, cbaoi in enumerate(column.value):
                    column.value[i] = cbaoi.remove_out_of_image_fraction_(
                        self.fraction)

        return batch

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.fraction]


class ClipCBAsToImagePlanes(Augmenter):
    """Clip coordinate-based augmentables to areas within the image plane.

    This augmenter inspects all coordinate-based augmentables (e.g.
    bounding boxes, line strings) within a given batch and from each of them
    parts that are outside of the image plane. Parts within the image plane
    will be retained. This may e.g. shrink down bounding boxes. For keypoints,
    it removes any single points outside of the image plane. Any augmentable
    that is completely outside of the image plane will be removed.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; fully tested
        * ``uint32``: yes; fully tested
        * ``uint64``: yes; fully tested
        * ``int8``: yes; fully tested
        * ``int16``: yes; fully tested
        * ``int32``: yes; fully tested
        * ``int64``: yes; fully tested
        * ``float16``: yes; fully tested
        * ``float32``: yes; fully tested
        * ``float64``: yes; fully tested
        * ``float128``: yes; fully tested
        * ``bool``: yes; fully tested

    Parameters
    ----------
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
    >>> aug = iaa.Sequential([
    >>>     iaa.Affine(translate_px={"x": (-100, 100)}),
    >>>     iaa.ClipCBAsToImagePlanes()
    >>> ])

    Translate input data on the x-axis by ``-100`` to ``100`` pixels,
    then cut all coordinate-based augmentables (e.g. bounding boxes) down
    to areas that are within the image planes of their corresponding images.

    """

    # Added in 0.4.0.
    def __init__(self,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(ClipCBAsToImagePlanes, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        for column in batch.columns:
            if column.name in ["keypoints", "bounding_boxes", "polygons",
                               "line_strings"]:
                for i, cbaoi in enumerate(column.value):
                    column.value[i] = cbaoi.clip_out_of_image_()

        return batch

    # Added in 0.4.0.
    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []
