"""
Augmenters that apply changes to images based on segmentation methods.

List of augmenters:

    * :class:`Superpixels`
    * :class:`Voronoi`
    * :class:`UniformVoronoi`
    * :class:`RegularGridVoronoi`
    * :class:`RelativeRegularGridVoronoi`

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod

import numpy as np
# use skimage.segmentation instead `from skimage import segmentation` here,
# because otherwise unittest seems to mix up imgaug.augmenters.segmentation
# with skimage.segmentation for whatever reason
import skimage.segmentation
import skimage.measure
import six
import six.moves as sm

import imgaug as ia
from . import meta
from .. import random as iarandom
from .. import parameters as iap
from .. import dtypes as iadt


# TODO merge this into imresize?
def _ensure_image_max_size(image, max_size, interpolation):
    """Ensure that images do not exceed a required maximum sidelength.

    This downscales to `max_size` if any side violates that maximum.
    The other side is downscaled too so that the aspect ratio is maintained.

    **Supported dtypes**:

    See :func:`~imgaug.imgaug.imresize_single_image`.

    Parameters
    ----------
    image : ndarray
        Image to potentially downscale.

    max_size : int
        Maximum length of any side of the image.

    interpolation : string or int
        See :func:`~imgaug.imgaug.imresize_single_image`.

    """
    if max_size is not None:
        size = max(image.shape[0], image.shape[1])
        if size > max_size:
            resize_factor = max_size / size
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            image = ia.imresize_single_image(
                image,
                (new_height, new_width),
                interpolation=interpolation)
    return image


# TODO add compactness parameter
class Superpixels(meta.Augmenter):
    """Transform images parially/completely to their superpixel representation.

    This implementation uses skimage's version of the SLIC algorithm.

    .. note::

        This augmenter is fairly slow. See :ref:`performance`.

    **Supported dtypes**:

    if (image size <= max_size):

        * ``uint8``: yes; fully tested
        * ``uint16``: yes; tested
        * ``uint32``: yes; tested
        * ``uint64``: limited (1)
        * ``int8``: yes; tested
        * ``int16``: yes; tested
        * ``int32``: yes; tested
        * ``int64``: limited (1)
        * ``float16``: no (2)
        * ``float32``: no (2)
        * ``float64``: no (3)
        * ``float128``: no (2)
        * ``bool``: yes; tested

        - (1) Superpixel mean intensity replacement requires computing
              these means as ``float64`` s. This can cause inaccuracies for
              large integer values.
        - (2) Error in scikit-image.
        - (3) Loss of resolution in scikit-image.

    if (image size > max_size):

        minimum of (
            ``imgaug.augmenters.segmentation.Superpixels(image size <= max_size)``,
            :func:`~imgaug.augmenters.segmentation._ensure_image_max_size`
        )

    Parameters
    ----------
    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    n_segments : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Rough target number of how many superpixels to generate (the algorithm
        may deviate from this number). Lower value will lead to coarser
        superpixels. Higher values are computationally more intensive and
        will hence lead to a slowdown.

            * If a single ``int``, then that value will always be used as the
              number of segments.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.Superpixels(p_replace=1.0, n_segments=64)

    Generate around ``64`` superpixels per image and replace all of them with
    their average color (standard superpixel image).

    >>> aug = iaa.Superpixels(p_replace=0.5, n_segments=64)

    Generate around ``64`` superpixels per image and replace half of them
    with their average color, while the other half are left unchanged (i.e.
    they still show the input image's content).

    >>> aug = iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))

    Generate between ``16`` and ``128`` superpixels per image and replace
    ``25`` to ``100`` percent of them with their average color.

    """

    def __init__(self, p_replace=(0.5, 1.0), n_segments=(50, 120),
                 max_size=128, interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Superpixels, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True)
        self.n_segments = iap.handle_discrete_param(
            n_segments, "n_segments", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.max_size = max_size
        self.interpolation = interpolation

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(images,
                         allowed=["bool",
                                  "uint8", "uint16", "uint32", "uint64",
                                  "int8", "int16", "int32", "int64"],
                         disallowed=["uint128", "uint256",
                                     "int128", "int256",
                                     "float16", "float32", "float64",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        nb_images = len(images)
        rss = random_state.duplicate(1+nb_images)
        n_segments_samples = self.n_segments.draw_samples(
            (nb_images,), random_state=rss[0])

        # We cant reduce images to 0 or less segments, hence we pick the
        # lowest possible value in these cases (i.e. 1). The alternative
        # would be to not perform superpixel detection in these cases
        # (akin to n_segments=#pixels).
        # TODO add test for this
        n_segments_samples = np.clip(n_segments_samples, 1, None)

        for i, (image, rs) in enumerate(zip(images, rss[1:])):
            if image.size == 0:
                # Image with 0-sized axis, nothing to change.
                # Placing this before the sampling step should be fine.
                continue

            replace_samples = self.p_replace.draw_samples(
                (n_segments_samples[i],), random_state=rs)

            if np.max(replace_samples) == 0:
                # not a single superpixel would be replaced by its average
                # color, i.e. the image would not be changed, so just keep it
                continue

            orig_shape = image.shape
            image = _ensure_image_max_size(image, self.max_size,
                                           self.interpolation)

            segments = skimage.segmentation.slic(
                image, n_segments=n_segments_samples[i], compactness=10)

            image_aug = self._replace_segments(image, segments, replace_samples)

            if orig_shape != image_aug.shape:
                image_aug = ia.imresize_single_image(
                    image_aug,
                    orig_shape[0:2],
                    interpolation=self.interpolation)

            batch.images[i] = image_aug
        return batch

    @classmethod
    def _replace_segments(cls, image, segments, replace_samples):
        min_value, _center_value, max_value = \
                iadt.get_value_range_of_dtype(image.dtype)
        image_sp = np.copy(image)

        nb_channels = image.shape[2]
        for c in sm.xrange(nb_channels):
            # segments+1 here because otherwise regionprops always
            # misses the last label
            regions = skimage.measure.regionprops(
                segments+1, intensity_image=image[..., c])
            for ridx, region in enumerate(regions):
                # with mod here, because slic can sometimes create more
                # superpixel than requested. replace_samples then does not
                # have enough values, so we just start over with the first one
                # again.
                if replace_samples[ridx % len(replace_samples)] > 0.5:
                    mean_intensity = region.mean_intensity
                    image_sp_c = image_sp[..., c]

                    if image_sp_c.dtype.kind in ["i", "u", "b"]:
                        # After rounding the value can end up slightly outside
                        # of the value_range. Hence, we need to clip. We do
                        # clip via min(max(...)) instead of np.clip because
                        # the latter one does not seem to keep dtypes for
                        # dtypes with large itemsizes (e.g. uint64).
                        value = int(np.round(mean_intensity))
                        value = min(max(value, min_value), max_value)
                    else:
                        value = mean_intensity

                    image_sp_c[segments == ridx] = value

        return image_sp

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p_replace, self.n_segments, self.max_size,
                self.interpolation]


# TODO don't average the alpha channel for RGBA?
def segment_voronoi(image, cell_coordinates, replace_mask=None):
    """Average colors within voronoi cells of an image.

    Parameters
    ----------
    image : ndarray
        The image to convert to a voronoi image. May be ``HxW`` or
        ``HxWxC``. Note that for ``RGBA`` images the alpha channel
        will currently also by averaged.

    cell_coordinates : ndarray
        A ``Nx2`` float array containing the center coordinates of voronoi
        cells on the image. Values are expected to be in the interval
        ``[0.0, height-1.0]`` for the y-axis (x-axis analogous).
        If this array contains no coordinate, the image will not be
        changed.

    replace_mask : None or ndarray, optional
        Boolean mask of the same length as `cell_coordinates`, denoting
        for each cell whether its pixels are supposed to be replaced
        by the cell's average color (``True``) or left untouched (``False``).
        If this is set to ``None``, all cells will be replaced.

    Returns
    -------
    ndarray
        Voronoi image.

    """
    input_dims = image.ndim
    if input_dims == 2:
        image = image[..., np.newaxis]

    if len(cell_coordinates) <= 0:
        if input_dims == 2:
            return image[..., 0]
        return image

    height, width = image.shape[0:2]
    pixel_coords, ids_of_nearest_cells = \
        _match_pixels_with_voronoi_cells(height, width, cell_coordinates)
    cell_colors = _compute_avg_segment_colors(
        image, pixel_coords, ids_of_nearest_cells,
        len(cell_coordinates))

    image_aug = _render_segments(image, ids_of_nearest_cells, cell_colors,
                                 replace_mask)

    if input_dims == 2:
        return image_aug[..., 0]
    return image_aug


def _match_pixels_with_voronoi_cells(height, width, cell_coordinates):
    # deferred import so that scipy is an optional dependency
    from scipy.spatial import cKDTree as KDTree  # TODO add scipy for reqs
    tree = KDTree(cell_coordinates)
    pixel_coords = _generate_pixel_coords(height, width)
    pixel_coords_subpixel = pixel_coords.astype(np.float32) + 0.5
    ids_of_nearest_cells = tree.query(pixel_coords_subpixel)[1]
    return pixel_coords, ids_of_nearest_cells


def _generate_pixel_coords(height, width):
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    return np.c_[xx.ravel(), yy.ravel()]


def _compute_avg_segment_colors(image, pixel_coords, ids_of_nearest_segments,
                                nb_segments):
    nb_channels = image.shape[2]
    cell_colors = np.zeros((nb_segments, nb_channels), dtype=np.float64)
    cell_counters = np.zeros((nb_segments,), dtype=np.uint32)

    # TODO vectorize
    for pixel_coord, id_of_nearest_cell in zip(pixel_coords,
                                               ids_of_nearest_segments):
        # pixel_coord is (x,y), so we have to swap it to access the HxW image
        pixel_coord_yx = pixel_coord[::-1]
        cell_colors[id_of_nearest_cell] += image[tuple(pixel_coord_yx)]
        cell_counters[id_of_nearest_cell] += 1

    # cells without associated pixels can have a count of 0, we clip
    # here to 1 as the result for these cells doesn't matter
    cell_counters = np.clip(cell_counters, 1, None)

    cell_colors = cell_colors / cell_counters[:, np.newaxis]

    return cell_colors.astype(np.uint8)


def _render_segments(image, ids_of_nearest_segments, avg_segment_colors,
                     replace_mask):
    ids_of_nearest_segments = np.copy(ids_of_nearest_segments)
    height, width, nb_channels = image.shape

    # without replace_mask we could reduce this down to:
    # data = cell_colors[ids_of_nearest_cells, :].reshape(
    #     (width, height, 3))
    # data = np.transpose(data, (1, 0, 2))

    keep_mask = (~replace_mask) if replace_mask is not None else None
    if keep_mask is None or not np.any(keep_mask):
        data = avg_segment_colors[ids_of_nearest_segments, :]
    else:
        ids_to_keep = np.nonzero(keep_mask)[0]
        indices_to_keep = np.where(
            np.isin(ids_of_nearest_segments, ids_to_keep))[0]
        data = avg_segment_colors[ids_of_nearest_segments, :]

        image_data = image.reshape((height*width, -1))
        data[indices_to_keep] = image_data[indices_to_keep, :]
    data = data.reshape((height, width, nb_channels))
    return data


# TODO this can be reduced down to a similar problem as Superpixels:
#      generate an integer-based class id map of segments, then replace all
#      segments with the same class id by the average color within that
#      segment
class Voronoi(meta.Augmenter):
    """Average colors of an image within Voronoi cells.

    This augmenter performs the following steps:

        1. Query `points_sampler` to sample random coordinates of cell
           centers. On the image.
        2. Estimate for each pixel to which voronoi cell (i.e. segment)
           it belongs. Each pixel belongs to the cell with the closest center
           coordinate (euclidean distance).
        3. Compute for each cell the average color of the pixels within it.
        4. Replace the pixels of `p_replace` percent of all cells by their
           average color. Do not change the pixels of ``(1 - p_replace)``
           percent of all cells. (The percentages are average values over
           many images. Some images may get more/less cells replaced by
           their average color.)

    This code is very loosely based on
    https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map/50345#50345

    **Supported dtypes**:

    if (image size <= max_size):

        * ``uint8``: yes; fully tested
        * ``uint16``: no; not tested
        * ``uint32``: no; not tested
        * ``uint64``: no; not tested
        * ``int8``: no; not tested
        * ``int16``: no; not tested
        * ``int32``: no; not tested
        * ``int64``: no; not tested
        * ``float16``: no; not tested
        * ``float32``: no; not tested
        * ``float64``: no; not tested
        * ``float128``: no; not tested
        * ``bool``: no; not tested

    if (image size > max_size):

        minimum of (
            ``imgaug.augmenters.segmentation.Voronoi(image size <= max_size)``,
            :func:`~imgaug.augmenters.segmentation._ensure_image_max_size`
        )

    Parameters
    ----------
    points_sampler : IPointsSampler
        A points sampler which will be queried per image to generate the
        coordinates of the centers of voronoi cells.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> points_sampler = iaa.RegularGridPointsSampler(n_cols=20, n_rows=40)
    >>> aug = iaa.Voronoi(points_sampler)

    Create an augmenter that places a ``20x40`` (``HxW``) grid of cells on
    the image and replaces all pixels within each cell by the cell's average
    color. The process is performed at an image size not exceeding ``128`` px
    on any side (default). If necessary, the downscaling is performed using
    ``linear`` interpolation (default).

    >>> points_sampler = iaa.DropoutPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(
    >>>         n_cols_frac=(0.05, 0.2),
    >>>         n_rows_frac=0.1),
    >>>     0.2)
    >>> aug = iaa.Voronoi(points_sampler, p_replace=0.9, max_size=None)

    Create a voronoi augmenter that generates a grid of cells dynamically
    adapted to the image size. Larger images get more cells. On the x-axis,
    the distance between two cells is ``w * W`` pixels, where ``W`` is the
    width of the image and ``w`` is always ``0.1``. On the y-axis,
    the distance between two cells is ``h * H`` pixels, where ``H`` is the
    height of the image and ``h`` is sampled uniformly from the interval
    ``[0.05, 0.2]``. To make the voronoi pattern less regular, about ``20``
    percent of the cell coordinates are randomly dropped (i.e. the remaining
    cells grow in size). In contrast to the first example, the image is not
    resized (if it was, the sampling would happen *after* the resizing,
    which would affect ``W`` and ``H``). Not all voronoi cells are replaced
    by their average color, only around ``90`` percent of them. The
    remaining ``10`` percent's pixels remain unchanged.

    """

    def __init__(self, points_sampler, p_replace=1.0, max_size=128,
                 interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(Voronoi, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

        assert isinstance(points_sampler, IPointsSampler), (
            "Expected 'points_sampler' to be an instance of IPointsSampler, "
            "got %s." % (type(points_sampler),))
        self.points_sampler = points_sampler

        self.p_replace = iap.handle_probability_param(
            p_replace, "p_replace", tuple_to_uniform=True, list_to_choice=True)

        self.max_size = max_size
        self.interpolation = interpolation

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        images = batch.images

        iadt.gate_dtypes(images,
                         allowed=["uint8"],
                         disallowed=["bool",
                                     "uint16", "uint32", "uint64", "uint128",
                                     "uint256",
                                     "int8", "int16", "int32", "int64",
                                     "int128", "int256",
                                     "float16", "float32", "float64",
                                     "float96", "float128", "float256"],
                         augmenter=self)

        rss = random_state.duplicate(len(images))
        for i, (image, rs) in enumerate(zip(images, rss)):
            batch.images[i] = self._augment_single_image(image, rs)
        return batch

    def _augment_single_image(self, image, random_state):
        rss = random_state.duplicate(2)
        orig_shape = image.shape
        image = _ensure_image_max_size(image, self.max_size, self.interpolation)

        cell_coordinates = self.points_sampler.sample_points([image], rss[0])[0]
        p_replace = self.p_replace.draw_samples((len(cell_coordinates),),
                                                rss[1])
        replace_mask = (p_replace > 0.5)

        image_aug = segment_voronoi(image, cell_coordinates, replace_mask)

        if orig_shape != image_aug.shape:
            image_aug = ia.imresize_single_image(
                image_aug,
                orig_shape[0:2],
                interpolation=self.interpolation)

        return image_aug

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.points_sampler, self.p_replace, self.max_size,
                self.interpolation]


class UniformVoronoi(Voronoi):
    """Uniformly sample Voronoi cells on images and average colors within them.

    This augmenter is a shortcut for the combination of
    :class:`~imgaug.augmenters.segmentation.Voronoi` with
    :class:`~imgaug.augmenters.segmentation.UniformPointsSampler`. Hence, it
    generates a fixed amount of ``N`` random coordinates of voronoi cells on
    each image. The cell coordinates are sampled uniformly using the image
    height and width as maxima.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.segmentation.Voronoi`.

    Parameters
    ----------
    n_points : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of points to sample on each image.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.UniformVoronoi((100, 500))

    Sample for each image uniformly the number of voronoi cells ``N`` from the
    interval ``[100, 500]``. Then generate ``N`` coordinates by sampling
    uniformly the x-coordinates from ``[0, W]`` and the y-coordinates from
    ``[0, H]``, where ``H`` is the image height and ``W`` the image width.
    Then use these coordinates to group the image pixels into voronoi
    cells and average the colors within them. The process is performed at an
    image size not exceeding ``128`` px on any side (default). If necessary,
    the downscaling is performed using ``linear`` interpolation (default).

    >>> aug = iaa.UniformVoronoi(250, p_replace=0.9, max_size=None)

    Same as above, but always samples ``N=250`` cells, replaces only
    ``90`` percent of them with their average color (the pixels of the
    remaining ``10`` percent are not changed) and performs the transformation
    at the original image size (``max_size=None``).

    """

    def __init__(self, n_points=(50, 500), p_replace=(0.5, 1.0), max_size=128,
                 interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(UniformVoronoi, self).__init__(
            points_sampler=UniformPointsSampler(n_points),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RegularGridVoronoi(Voronoi):
    """Sample Voronoi cells from regular grids and color-average them.

    This augmenter is a shortcut for the combination of
    :class:`~imgaug.augmenters.segmentation.Voronoi`,
    :class:`~imgaug.augmenters.segmentation.RegularGridPointsSampler` and
    :class:`~imgaug.augmenters.segmentation.DropoutPointsSampler`. Hence, it
    generates a regular grid with ``R`` rows and ``C`` columns of coordinates
    on each image. Then, it drops ``p`` percent of the ``R*C`` coordinates
    to randomize the grid. Each image pixel then belongs to the voronoi
    cell with the closest coordinate.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.segmentation.Voronoi`.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of coordinates to place on each image, i.e. the number
        of coordinates on the y-axis. Note that for each image, the sampled
        value is clipped to the interval ``[1..H]``, where ``H`` is the image
        height.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of columns of coordinates to place on each image, i.e. the
        number of coordinates on the x-axis. Note that for each image, the
        sampled value is clipped to the interval ``[1..W]``, where ``W`` is
        the image width.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    p_drop_points : number or tuple of number or imgaug.parameters.StochasticParameter, optional
        The probability that a coordinate will be removed from the list
        of all sampled coordinates. A value of ``1.0`` would mean that (on
        average) ``100`` percent of all coordinates will be dropped,
        while ``0.0`` denotes ``0`` percent. Note that this sampler will
        always ensure that at least one coordinate is left after the dropout
        operation, i.e. even ``1.0`` will only drop all *except one*
        coordinate.

            * If a ``float``, then that value will be used for all images.
            * If a ``tuple`` ``(a, b)``, then a value ``p`` will be sampled
              from the interval ``[a, b]`` per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per coordinate whether it should be *kept* (sampled
              value of ``>0.5``) or shouldn't be kept (sampled value of
              ``<=0.5``). If you instead want to provide the probability as
              a stochastic parameter, you can usually do
              ``imgaug.parameters.Binomial(1-p)`` to convert parameter `p` to
              a 0/1 representation.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that number will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.RegularGridVoronoi(10, 20)

    Place a regular grid of ``10x20`` (``height x width``) coordinates on
    each image. Randomly drop on average ``20`` percent of these points
    to create a less regular pattern. Then use the remaining coordinates
    to group the image pixels into voronoi cells and average the colors
    within them. The process is performed at an image size not exceeding
    ``128`` px on any side (default). If necessary, the downscaling is
    performed using ``linear`` interpolation (default).

    >>> aug = iaa.RegularGridVoronoi(
    >>>     (10, 30), 20, p_drop_points=0.0, p_replace=0.9, max_size=None)

    Same as above, generates a grid with randomly ``10`` to ``30`` rows,
    drops none of the generates points, replaces only ``90`` percent of
    the voronoi cells with their average color (the pixels of the remaining
    ``10`` percent are not changed) and performs the transformation
    at the original image size (``max_size=None``).

    """

    def __init__(self, n_rows=(10, 30), n_cols=(10, 30),
                 p_drop_points=(0.0, 0.5), p_replace=(0.5, 1.0),
                 max_size=128, interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(RegularGridVoronoi, self).__init__(
            points_sampler=DropoutPointsSampler(
                RegularGridPointsSampler(n_rows, n_cols),
                p_drop_points
            ),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


class RelativeRegularGridVoronoi(Voronoi):
    """Sample Voronoi cells from image-dependent grids and color-average them.

    This augmenter is a shortcut for the combination of
    :class:`~imgaug.augmenters.segmentation.Voronoi`,
    :class:`~imgaug.augmenters.segmentation.RegularGridPointsSampler` and
    :class:`~imgaug.augmenters.segmentation.DropoutPointsSampler`. Hence, it
    generates a regular grid with ``R`` rows and ``C`` columns of coordinates
    on each image. Then, it drops ``p`` percent of the ``R*C`` coordinates
    to randomize the grid. Each image pixel then belongs to the voronoi
    cell with the closest coordinate.

    .. note::

        In contrast to the other voronoi augmenters, this one uses
        ``None`` as the default value for `max_size`, i.e. the color averaging
        is always performed at full resolution. This enables the augmenter to
        make use of the additional points on larger images. It does
        however slow down the augmentation process.

    **Supported dtypes**:

    See :class:`~imgaug.augmenters.segmentation.Voronoi`.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the y-axis. For a value
        ``y`` and image height ``H`` the number of actually placed coordinates
        (i.e. computed rows) is given by ``int(round(y*H))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,H]``, where ``H`` is the image height.

            * If a single ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the x-axis. For a value
        ``x`` and image height ``W`` the number of actually placed coordinates
        (i.e. computed columns) is given by ``int(round(x*W))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,W]``, where ``W`` is the image width.

            * If a single ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    p_drop_points : number or tuple of number or imgaug.parameters.StochasticParameter, optional
        The probability that a coordinate will be removed from the list
        of all sampled coordinates. A value of ``1.0`` would mean that (on
        average) ``100`` percent of all coordinates will be dropped,
        while ``0.0`` denotes ``0`` percent. Note that this sampler will
        always ensure that at least one coordinate is left after the dropout
        operation, i.e. even ``1.0`` will only drop all *except one*
        coordinate.

            * If a ``float``, then that value will be used for all images.
            * If a ``tuple`` ``(a, b)``, then a value ``p`` will be sampled
              from the interval ``[a, b]`` per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per coordinate whether it should be *kept* (sampled
              value of ``>0.5``) or shouldn't be kept (sampled value of
              ``<=0.5``). If you instead want to provide the probability as
              a stochastic parameter, you can usually do
              ``imgaug.parameters.Binomial(1-p)`` to convert parameter `p` to
              a 0/1 representation.

    p_replace : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Defines for any segment the probability that the pixels within that
        segment are replaced by their average color (otherwise, the pixels
        are not changed).
        Examples:

            * A probability of ``0.0`` would mean, that the pixels in no
              segment are replaced by their average color (image is not
              changed at all).
            * A probability of ``0.5`` would mean, that around half of all
              segments are replaced by their average color.
            * A probability of ``1.0`` would mean, that all segments are
              replaced by their average color (resulting in a voronoi
              image).

        Behaviour based on chosen datatypes for this parameter:

            * If a ``number``, then that ``number`` will always be used.
            * If ``tuple`` ``(a, b)``, then a random probability will be
              sampled from the interval ``[a, b]`` per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, it is expected to return
              values between ``0.0`` and ``1.0`` and will be queried *for each
              individual segment* to determine whether it is supposed to
              be averaged (``>0.5``) or not (``<=0.5``).
              Recommended to be some form of ``Binomial(...)``.

    max_size : int or None, optional
        Maximum image size at which the augmentation is performed.
        If the width or height of an image exceeds this value, it will be
        downscaled before the augmentation so that the longest side
        matches `max_size`.
        This is done to speed up the process. The final output image has the
        same size as the input image. Note that in case `p_replace` is below
        ``1.0``, the down-/upscaling will affect the not-replaced pixels too.
        Use ``None`` to apply no down-/upscaling.

    interpolation : int or str, optional
        Interpolation method to use during downscaling when `max_size` is
        exceeded. Valid methods are the same as in
        :func:`~imgaug.imgaug.imresize_single_image`.

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
    >>> aug = iaa.RelativeRegularGridVoronoi(0.1, 0.25)

    Place a regular grid of ``R x C`` coordinates on each image, where
    ``R`` is the number of rows and computed as ``R=0.1*H`` with ``H`` being
    the height of the input image. ``C`` is the number of columns and
    analogously estimated from the image width ``W`` as ``C=0.25*W``.
    Larger images will lead to larger ``R`` and ``C`` values.
    On average, ``20`` percent of these grid coordinates are randomly
    dropped to create a less regular pattern. Then, the remaining coordinates
    are used to group the image pixels into voronoi cells and the colors
    within them are averaged.

    >>> aug = iaa.RelativeRegularGridVoronoi(
    >>>     (0.03, 0.1), 0.1, p_drop_points=0.0, p_replace=0.9, max_size=512)

    Same as above, generates a grid with randomly ``R=r*H`` rows, where
    ``r`` is sampled uniformly from the interval ``[0.03, 0.1]`` and
    ``C=0.1*W`` rows. No points are dropped. The augmenter replaces only
    ``90`` percent of the voronoi cells with their average color (the pixels
    of the remaining ``10`` percent are not changed). Images larger than
    ``512`` px are temporarily downscaled (*before* sampling the grid points)
    so that no side exceeds ``512`` px. This improves performance, but
    degrades the quality of the resulting image.

    """

    def __init__(self, n_rows_frac=(0.05, 0.15), n_cols_frac=(0.05, 0.15),
                 p_drop_points=(0.0, 0.5), p_replace=(0.5, 1.0),
                 max_size=None, interpolation="linear",
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(RelativeRegularGridVoronoi, self).__init__(
            points_sampler=DropoutPointsSampler(
                RelativeRegularGridPointsSampler(n_rows_frac, n_cols_frac),
                p_drop_points
            ),
            p_replace=p_replace,
            max_size=max_size,
            interpolation=interpolation,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)


@six.add_metaclass(ABCMeta)
class IPointsSampler(object):
    """Interface for all point samplers.

    Point samplers return coordinate arrays of shape ``Nx2``.
    These coordinates can be used in other augmenters, see e.g.
    :class:`~imgaug.augmenters.segmentation.Voronoi`.

    """

    @abstractmethod
    def sample_points(self, images, random_state):
        """Generate coordinates of points on images.

        Parameters
        ----------
        images : ndarray or list of ndarray
            One or more images for which to generate points.
            If this is a ``list`` of arrays, each one of them is expected to
            have three dimensions.
            If this is an array, it must be four-dimensional and the first
            axis is expected to denote the image index. For ``RGB`` images
            the array would hence have to be of shape ``(N, H, W, 3)``.

        random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState
            A random state to use for any probabilistic function required
            during the point sampling.
            See :func:`~imgaug.random.RNG` for details.

        Returns
        -------
        ndarray
            An ``(N,2)`` ``float32`` array containing ``(x,y)`` subpixel
            coordinates, all of which being within the intervals
            ``[0.0, width]`` and ``[0.0, height]``.

        """


def _verify_sample_points_images(images):
    assert len(images) > 0, "Expected at least one image, got zero."
    if isinstance(images, list):
        assert all([ia.is_np_array(image) for image in images]), (
            "Expected list of numpy arrays, got list of types %s." % (
                ", ".join([str(type(image)) for image in images]),))
        assert all([image.ndim == 3 for image in images]), (
            "Expected each image to have three dimensions, "
            "got dimensions %s." % (
                ", ".join([str(image.ndim) for image in images]),))
    else:
        assert ia.is_np_array(images), (
            "Expected either a list of numpy arrays or a single numpy "
            "array of shape NxHxWxC. Got type %s." % (type(images),))
        assert images.ndim == 4, (
            "Expected a four-dimensional array of shape NxHxWxC. "
            "Got shape %d dimensions (shape: %s)." % (
                images.ndim, images.shape))


class RegularGridPointsSampler(IPointsSampler):
    """Sampler that generates a regular grid of coordinates on an image.

    'Regular grid' here means that on each axis all coordinates have the
    same distance from each other. Note that the distance may change between
    axis.

    Parameters
    ----------
    n_rows : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of rows of coordinates to place on each image, i.e. the number
        of coordinates on the y-axis. Note that for each image, the sampled
        value is clipped to the interval ``[1..H]``, where ``H`` is the image
        height.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of columns of coordinates to place on each image, i.e. the
        number of coordinates on the x-axis. Note that for each image, the
        sampled value is clipped to the interval ``[1..W]``, where ``W`` is
        the image width.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.RegularGridPointsSampler(
    >>>     n_rows=(5, 20),
    >>>     n_cols=50)

    Create a point sampler that generates regular grids of points. These grids
    contain ``r`` points on the y-axis, where ``r`` is sampled
    uniformly from the discrete interval ``[5..20]`` per image.
    On the x-axis, the grids always contain ``50`` points.

    """

    def __init__(self, n_rows, n_cols):
        self.n_rows = iap.handle_discrete_param(
            n_rows, "n_rows", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)
        self.n_cols = iap.handle_discrete_param(
            n_cols, "n_cols", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG(random_state)
        _verify_sample_points_images(images)

        n_rows_lst, n_cols_lst = self._draw_samples(images, random_state)
        return self._generate_point_grids(images, n_rows_lst, n_cols_lst)

    def _draw_samples(self, images, random_state):
        rss = random_state.duplicate(2)
        n_rows_lst = self.n_rows.draw_samples(len(images), random_state=rss[0])
        n_cols_lst = self.n_cols.draw_samples(len(images), random_state=rss[1])
        return self._clip_rows_and_cols(n_rows_lst, n_cols_lst, images)

    @classmethod
    def _clip_rows_and_cols(cls, n_rows_lst, n_cols_lst, images):
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])
        # We clip intentionally not to H-1 or W-1 here. If e.g. an image has
        # a width of 1, we want to get a maximum of 1 column of coordinates.
        # Note that we use two clips here instead of e.g. clip(., 1, height),
        # because images can have height/width zero and in these cases numpy
        # prefers the smaller value in clip(). But currently we want to get
        # at least 1 point for such images.
        n_rows_lst = np.clip(n_rows_lst, None, heights)
        n_cols_lst = np.clip(n_cols_lst, None, widths)
        n_rows_lst = np.clip(n_rows_lst, 1, None)
        n_cols_lst = np.clip(n_cols_lst, 1, None)
        return n_rows_lst, n_cols_lst

    @classmethod
    def _generate_point_grids(cls, images, n_rows_lst, n_cols_lst):
        grids = []
        for image, n_rows_i, n_cols_i in zip(images, n_rows_lst, n_cols_lst):
            grids.append(cls._generate_point_grid(image, n_rows_i, n_cols_i))
        return grids

    @classmethod
    def _generate_point_grid(cls, image, n_rows, n_cols):
        height, width = image.shape[0:2]

        # We do not have to subtract 1 here from height/width as these are
        # subpixel coordinates. Technically, we could also place the cell
        # centers outside of the image plane.
        y_spacing = height / n_rows
        y_start = 0.0 + y_spacing/2
        y_end = height - y_spacing/2
        if y_start - 1e-4 <= y_end <= y_start + 1e-4:
            yy = np.float32([y_start])
        else:
            yy = np.linspace(y_start, y_end, num=n_rows)

        x_spacing = width / n_cols
        x_start = 0.0 + x_spacing/2
        x_end = width - x_spacing/2
        if x_start - 1e-4 <= x_end <= x_start + 1e-4:
            xx = np.float32([x_start])
        else:
            xx = np.linspace(x_start, x_end, num=n_cols)

        xx, yy = np.meshgrid(xx, yy)
        grid = np.vstack([xx.ravel(), yy.ravel()]).T
        return grid

    def __repr__(self):
        return "RegularGridPointsSampler(%s, %s)" % (self.n_rows, self.n_cols)

    def __str__(self):
        return self.__repr__()


class RelativeRegularGridPointsSampler(IPointsSampler):
    """Regular grid coordinate sampler; places more points on larger images.

    This is similar to ``RegularGridPointsSampler``, but the number of rows
    and columns is given as fractions of each image's height and width.
    Hence, more coordinates are generated for larger images.

    Parameters
    ----------
    n_rows_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the y-axis. For a value
        ``y`` and image height ``H`` the number of actually placed coordinates
        (i.e. computed rows) is given by ``int(round(y*H))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,H]``, where ``H`` is the image height.

            * If a single ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    n_cols_frac : number or tuple of number or list of number or imgaug.parameters.StochasticParameter, optional
        Relative number of coordinates to place on the x-axis. For a value
        ``x`` and image height ``W`` the number of actually placed coordinates
        (i.e. computed columns) is given by ``int(round(x*W))``.
        Note that for each image, the number of coordinates is clipped to the
        interval ``[1,W]``, where ``W`` is the image width.

            * If a single ``number``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the interval
              ``[a, b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.RelativeRegularGridPointsSampler(
    >>>     n_rows_frac=(0.01, 0.1),
    >>>     n_cols_frac=0.2)

    Create a point sampler that generates regular grids of points. These grids
    contain ``round(y*H)`` points on the y-axis, where ``y`` is sampled
    uniformly from the interval ``[0.01, 0.1]`` per image and ``H`` is the
    image height. On the x-axis, the grids always contain ``0.2*W`` points,
    where ``W`` is the image width.

    """

    def __init__(self, n_rows_frac, n_cols_frac):
        eps = 1e-4
        self.n_rows_frac = iap.handle_continuous_param(
            n_rows_frac, "n_rows_frac", value_range=(0.0+eps, 1.0),
            tuple_to_uniform=True, list_to_choice=True)
        self.n_cols_frac = iap.handle_continuous_param(
            n_cols_frac, "n_cols_frac", value_range=(0.0+eps, 1.0),
            tuple_to_uniform=True, list_to_choice=True)

    def sample_points(self, images, random_state):
        # pylint: disable=protected-access
        random_state = iarandom.RNG(random_state)
        _verify_sample_points_images(images)

        n_rows, n_cols = self._draw_samples(images, random_state)
        return RegularGridPointsSampler._generate_point_grids(images,
                                                              n_rows, n_cols)

    def _draw_samples(self, images, random_state):
        # pylint: disable=protected-access
        n_augmentables = len(images)
        rss = random_state.duplicate(2)
        n_rows_frac = self.n_rows_frac.draw_samples(n_augmentables,
                                                    random_state=rss[0])
        n_cols_frac = self.n_cols_frac.draw_samples(n_augmentables,
                                                    random_state=rss[1])
        heights = np.int32([image.shape[0] for image in images])
        widths = np.int32([image.shape[1] for image in images])

        n_rows = np.round(n_rows_frac * heights)
        n_cols = np.round(n_cols_frac * widths)
        n_rows, n_cols = RegularGridPointsSampler._clip_rows_and_cols(
            n_rows, n_cols, images)

        return n_rows.astype(np.int32), n_cols.astype(np.int32)

    def __repr__(self):
        return "RelativeRegularGridPointsSampler(%s, %s)" % (
            self.n_rows_frac, self.n_cols_frac)

    def __str__(self):
        return self.__repr__()


class DropoutPointsSampler(IPointsSampler):
    """Remove a defined fraction of sampled points.

    Parameters
    ----------
    other_points_sampler : IPointsSampler
        Another point sampler that is queried to generate a list of points.
        The dropout operation will be applied to that list.

    p_drop : number or tuple of number or imgaug.parameters.StochasticParameter
        The probability that a coordinate will be removed from the list
        of all sampled coordinates. A value of ``1.0`` would mean that (on
        average) ``100`` percent of all coordinates will be dropped,
        while ``0.0`` denotes ``0`` percent. Note that this sampler will
        always ensure that at least one coordinate is left after the dropout
        operation, i.e. even ``1.0`` will only drop all *except one*
        coordinate.

            * If a ``float``, then that value will be used for all images.
            * If a ``tuple`` ``(a, b)``, then a value ``p`` will be sampled
              from the interval ``[a, b]`` per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              determine per coordinate whether it should be *kept* (sampled
              value of ``>0.5``) or shouldn't be kept (sampled value of
              ``<=0.5``). If you instead want to provide the probability as
              a stochastic parameter, you can usually do
              ``imgaug.parameters.Binomial(1-p)`` to convert parameter `p` to
              a 0/1 representation.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.DropoutPointsSampler(
    >>>     iaa.RegularGridPointsSampler(10, 20),
    >>>     0.2)

    Create a point sampler that first generates points following a regular
    grid of ``10`` rows and ``20`` columns, then randomly drops ``20`` percent
    of these points.

    """

    def __init__(self, other_points_sampler, p_drop):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_points_sampler', got type %s." % (
                type(other_points_sampler),))
        self.other_points_sampler = other_points_sampler
        self.p_drop = self._convert_p_drop_to_inverted_mask_param(p_drop)

    @classmethod
    def _convert_p_drop_to_inverted_mask_param(cls, p_drop):
        # TODO this is the same as in Dropout, make DRY
        # TODO add list as an option
        if ia.is_single_number(p_drop):
            p_drop = iap.Binomial(1 - p_drop)
        elif ia.is_iterable(p_drop):
            assert len(p_drop) == 2, (
                "Expected 'p_drop' given as an iterable to contain exactly "
                "2 values, got %d." % (len(p_drop),))
            assert p_drop[0] < p_drop[1], (
                "Expected 'p_drop' given as iterable to contain exactly 2 "
                "values (a, b) with a < b. Got %.4f and %.4f." % (
                    p_drop[0], p_drop[1]))
            assert 0 <= p_drop[0] <= 1.0 and 0 <= p_drop[1] <= 1.0, (
                "Expected 'p_drop' given as iterable to only contain values "
                "in the interval [0.0, 1.0], got %.4f and %.4f." % (
                    p_drop[0], p_drop[1]))
            p_drop = iap.Binomial(iap.Uniform(1 - p_drop[1], 1 - p_drop[0]))
        elif isinstance(p_drop, iap.StochasticParameter):
            pass
        else:
            raise Exception(
                "Expected p_drop to be float or int or StochasticParameter, "
                "got %s." % (type(p_drop),))
        return p_drop

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(2)
        points_on_images = self.other_points_sampler.sample_points(images,
                                                                   rss[0])
        drop_masks = self._draw_samples(points_on_images, rss[1])
        return self._apply_dropout_masks(points_on_images, drop_masks)

    def _draw_samples(self, points_on_images, random_state):
        rss = random_state.duplicate(len(points_on_images))
        drop_masks = [self._draw_samples_for_image(points_on_image, rs)
                      for points_on_image, rs
                      in zip(points_on_images, rss)]
        return drop_masks

    def _draw_samples_for_image(self, points_on_image, random_state):
        drop_samples = self.p_drop.draw_samples((len(points_on_image),),
                                                random_state)
        keep_mask = (drop_samples > 0.5)
        return keep_mask

    @classmethod
    def _apply_dropout_masks(cls, points_on_images, keep_masks):
        points_on_images_dropped = []
        for points_on_image, keep_mask in zip(points_on_images, keep_masks):
            if len(points_on_image) == 0:
                # other sampler didn't provide any points
                poi_dropped = points_on_image
            else:
                if not np.any(keep_mask):
                    # keep at least one point if all were supposed to be
                    # dropped
                    # TODO this could also be moved into its own point sampler,
                    #      like AtLeastOnePoint(...)
                    idx = (len(points_on_image) - 1) // 2
                    keep_mask = np.copy(keep_mask)
                    keep_mask[idx] = True
                poi_dropped = points_on_image[keep_mask, :]
            points_on_images_dropped.append(poi_dropped)
        return points_on_images_dropped

    def __repr__(self):
        return "DropoutPointsSampler(%s, %s)" % (self.other_points_sampler,
                                                 self.p_drop)

    def __str__(self):
        return self.__repr__()


class UniformPointsSampler(IPointsSampler):
    """Sample points uniformly on images.

    This point sampler generates `n_points` points per image. The x- and
    y-coordinates are both sampled from uniform distributions matching the
    respective image width and height.

    Parameters
    ----------
    n_points : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Number of points to sample on each image.

            * If a single ``int``, then that value will always be used.
            * If a ``tuple`` ``(a, b)``, then a value from the discrete
              interval ``[a..b]`` will be sampled per image.
            * If a ``list``, then a random value will be sampled from that
              ``list`` per image.
            * If a ``StochasticParameter``, then that parameter will be
              queried to draw one value per image.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.UniformPointsSampler(500)

    Create a point sampler that generates an array of ``500`` random points for
    each input image. The x- and y-coordinates of each point are sampled
    from uniform distributions.

    """

    def __init__(self, n_points):
        self.n_points = iap.handle_discrete_param(
            n_points, "n_points", value_range=(1, None),
            tuple_to_uniform=True, list_to_choice=True, allow_floats=False)

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(2)
        n_points_imagewise = self._draw_samples(len(images), rss[0])

        n_points_total = np.sum(n_points_imagewise)
        n_components_total = 2 * n_points_total
        coords_relative = rss[1].uniform(0.0, 1.0, n_components_total)
        coords_relative_xy = coords_relative.reshape(n_points_total, 2)

        return self._convert_relative_coords_to_absolute(
            coords_relative_xy, n_points_imagewise, images)

    def _draw_samples(self, n_augmentables, random_state):
        n_points = self.n_points.draw_samples((n_augmentables,),
                                              random_state=random_state)
        n_points_clipped = np.clip(n_points, 1, None)
        return n_points_clipped

    @classmethod
    def _convert_relative_coords_to_absolute(cls, coords_rel_xy,
                                             n_points_imagewise, images):
        coords_absolute = []
        i = 0
        for image, n_points_image in zip(images, n_points_imagewise):
            height, width = image.shape[0:2]
            xx = coords_rel_xy[i:i+n_points_image, 0]
            yy = coords_rel_xy[i:i+n_points_image, 1]

            xx_int = np.clip(np.round(xx * width), 0, width)
            yy_int = np.clip(np.round(yy * height), 0, height)

            coords_absolute.append(np.stack([xx_int, yy_int], axis=-1))
            i += n_points_image
        return coords_absolute

    def __repr__(self):
        return "UniformPointsSampler(%s)" % (self.n_points,)

    def __str__(self):
        return self.__repr__()


class SubsamplingPointsSampler(IPointsSampler):
    """Ensure that the number of sampled points is below a maximum.

    This point sampler will sample points from another sampler and
    then -- in case more points were generated than an allowed maximum --
    will randomly pick `n_points_max` of these.

    Parameters
    ----------
    other_points_sampler : IPointsSampler
        Another point sampler that is queried to generate a ``list`` of points.
        The dropout operation will be applied to that ``list``.

    n_points_max : int
        Maximum number of allowed points. If `other_points_sampler` generates
        more points than this maximum, a random subset of size `n_points_max`
        will be selected.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> sampler = iaa.SubsamplingPointsSampler(
    >>>     iaa.RelativeRegularGridPointsSampler(0.1, 0.2),
    >>>     50
    >>> )

    Create a points sampler that places ``y*H`` points on the y-axis (with
    ``y`` being ``0.1`` and ``H`` being an image's height) and ``x*W`` on
    the x-axis (analogous). Then, if that number of placed points exceeds
    ``50`` (can easily happen for larger images), a random subset of ``50``
    points will be picked and returned.

    """

    def __init__(self, other_points_sampler, n_points_max):
        assert isinstance(other_points_sampler, IPointsSampler), (
            "Expected to get an instance of IPointsSampler as argument "
            "'other_points_sampler', got type %s." % (
                type(other_points_sampler),))
        self.other_points_sampler = other_points_sampler
        self.n_points_max = np.clip(n_points_max, -1, None)
        if self.n_points_max == 0:
            ia.warn("Got n_points_max=0 in SubsamplingPointsSampler. "
                    "This will result in no points ever getting "
                    "returned.")

    def sample_points(self, images, random_state):
        random_state = iarandom.RNG(random_state)
        _verify_sample_points_images(images)

        rss = random_state.duplicate(len(images) + 1)
        points_on_images = self.other_points_sampler.sample_points(
            images, rss[-1])
        return [self._subsample(points_on_image, self.n_points_max, rs)
                for points_on_image, rs
                in zip(points_on_images, rss[:-1])]

    @classmethod
    def _subsample(cls, points_on_image, n_points_max, random_state):
        if len(points_on_image) <= n_points_max:
            return points_on_image
        indices = np.arange(len(points_on_image))
        indices_to_keep = random_state.permutation(indices)[0:n_points_max]
        return points_on_image[indices_to_keep]

    def __repr__(self):
        return "SubsamplingPointsSampler(%s, %d)" % (self.other_points_sampler,
                                                     self.n_points_max)

    def __str__(self):
        return self.__repr__()


# TODO Add points subsampler that drops points close to each other first
# TODO Add poisson points sampler
# TODO Add jitter points sampler that moves points around
# for both see https://codegolf.stackexchange.com/questions/50299/draw-an-image-as-a-voronoi-map/50345#50345
