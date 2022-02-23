"""Augmenters that help with debugging.

List of augmenters:

    * :class:`SaveDebugImageEveryNBatches`

Added in 0.4.0.

"""
from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty

import os
import collections

import six
import numpy as np
import imageio

import imgaug as ia
from .. import dtypes as iadt
from . import meta
from . import size as sizelib
from . import blend as blendlib

_COLOR_PINK = (255, 192, 203)
_COLOR_GRID_BACKGROUND = _COLOR_PINK


def _resizepad_to_size(image, size, cval):
    """Resize and pad and image to given size.

    This first resizes until one image size matches one size in `size` (while
    retaining the aspect ratio).
    Then it pads the other side until both sides match `size`.

    Added in 0.4.0.

    """
    # resize to height H and width W while keeping aspect ratio
    height = size[0]
    width = size[1]
    height_im = image.shape[0]
    width_im = image.shape[1]
    aspect_ratio_im = width_im / height_im

    # we know that height_im <= height and width_im <= width
    height_diff = height - height_im
    width_diff = width - width_im
    if height_diff < width_diff:
        height_im_rs = height
        width_im_rs = height * aspect_ratio_im
    else:
        height_im_rs = width / aspect_ratio_im
        width_im_rs = width

    height_im_rs = max(int(np.round(height_im_rs)), 1)
    width_im_rs = max(int(np.round(width_im_rs)), 1)

    image_rs = ia.imresize_single_image(image, (height_im_rs, width_im_rs))

    # pad to remaining size
    pad_y = height - height_im_rs
    pad_x = width - width_im_rs
    pad_top = int(np.floor(pad_y / 2))
    pad_right = int(np.ceil(pad_x / 2))
    pad_bottom = int(np.ceil(pad_y / 2))
    pad_left = int(np.floor(pad_x / 2))

    image_rs_pad = sizelib.pad(image_rs,
                               top=pad_top, right=pad_right,
                               bottom=pad_bottom, left=pad_left,
                               cval=cval)

    paddings = (pad_top, pad_right, pad_bottom, pad_left)
    return image_rs_pad, (height_im_rs, width_im_rs), paddings


# TODO rename to Grid
@six.add_metaclass(ABCMeta)
class _IDebugGridCell(object):
    """A single cell within a debug image's grid.

    Usually corresponds to one image, but can also be e.g. a title/description.

    Added in 0.4.0.

    """

    @abstractproperty
    def min_width(self):
        """Minimum width in pixels that the cell requires.

        Added in 0.4.0.

        """

    @abstractproperty
    def min_height(self):
        """Minimum height in pixels that the cell requires.

        Added in 0.4.0.

        """

    @abstractmethod
    def draw(self, height, width):
        """Draw the debug image grid cell's content.

        Added in 0.4.0.

        Parameters
        ----------
        height : int
            Expected height of the drawn cell image/array.

        width : int
            Expected width of the drawn cell image/array.

        Returns
        -------
        ndarray
            ``(H,W,3)`` Image.

        """


class _DebugGridBorderCell(_IDebugGridCell):
    """Helper to add a border around a cell within the debug image grid.

    Added in 0.4.0.

    """

    # Added in 0.4.0.
    def __init__(self, size, color, child):
        self.size = size
        self.color = color
        self.child = child

    # Added in 0.4.0.
    @property
    def min_height(self):
        return self.child.min_height

    # Added in 0.4.0.
    @property
    def min_width(self):
        return self.child.min_width

    # Added in 0.4.0.
    def draw(self, height, width):
        content = self.child.draw(height, width)
        content = sizelib.pad(content,
                              top=self.size, right=self.size,
                              bottom=self.size, left=self.size,
                              mode="constant", cval=self.color)
        return content


class _DebugGridTextCell(_IDebugGridCell):
    """Cell containing text.

    Added in 0.4.0.

    """

    # Added in 0.4.0.
    def __init__(self, text):
        self.text = text

    # Added in 0.4.0.
    @property
    def min_height(self):
        return max(20, len(self.text.split("\n")) * 17)

    # Added in 0.4.0.
    @property
    def min_width(self):
        lines = self.text.split("\n")
        if len(lines) == 0:
            return 20
        return max(20, int(7 * max([len(line) for line in lines])))

    # Added in 0.4.0.
    def draw(self, height, width):
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        image = ia.draw_text(image, 0, 0, self.text, color=(0, 0, 0),
                             size=12)
        return image


class _DebugGridImageCell(_IDebugGridCell):
    """Cell containing an image, possibly with an different-shaped overlay.

    Added in 0.4.0.

    """

    # Added in 0.4.0.
    def __init__(self, image, overlay=None, overlay_alpha=0.75):
        self.image = image
        self.overlay = overlay
        self.overlay_alpha = overlay_alpha

    # Added in 0.4.0.
    @property
    def min_height(self):
        return self.image.shape[0]

    # Added in 0.4.0.
    @property
    def min_width(self):
        return self.image.shape[1]

    # Added in 0.4.0.
    def draw(self, height, width):
        image = self.image
        kind = image.dtype.kind
        if kind == "b":
            image = image.astype(np.uint8) * 255
        elif kind == "u":
            min_value, _, max_value = iadt.get_value_range_of_dtype(image.dtype)
            image = image.astype(np.float64) / max_value
        elif kind == "i":
            min_value, _, max_value = iadt.get_value_range_of_dtype(image.dtype)
            dynamic_range = (max_value - min_value)
            image = (min_value + image.astype(np.float64)) / dynamic_range

        if image.dtype.kind == "f":
            image = (np.clip(image, 0, 1.0) * 255).astype(np.uint8)

        image_rsp, size_rs, paddings = _resizepad_to_size(
            image, (height, width), cval=_COLOR_GRID_BACKGROUND)

        blend = image_rsp
        if self.overlay is not None:
            overlay_rs = self._resize_overlay(self.overlay,
                                              image.shape[0:2])
            overlay_rsp = self._resize_overlay(overlay_rs, size_rs)
            overlay_rsp = sizelib.pad(overlay_rsp,
                                      top=paddings[0], right=paddings[1],
                                      bottom=paddings[2], left=paddings[3],
                                      cval=_COLOR_GRID_BACKGROUND)

            blend = blendlib.blend_alpha(overlay_rsp, image_rsp,
                                         alpha=self.overlay_alpha)

        return blend

    # Added in 0.4.0.
    @classmethod
    def _resize_overlay(cls, arr, size):
        arr_rs = ia.imresize_single_image(arr, size, interpolation="nearest")
        return arr_rs


class _DebugGridCBAsOICell(_IDebugGridCell):
    """Cell visualizing a coordinate-based augmentable.

    CBAsOI = coordinate-based augmentables on images,
    e.g. ``KeypointsOnImage``.

    Added in 0.4.0.

    """

    # Added in 0.4.0.
    def __init__(self, cbasoi, image):
        self.cbasoi = cbasoi
        self.image = image

    # Added in 0.4.0.
    @property
    def min_height(self):
        return self.image.shape[0]

    # Added in 0.4.0.
    @property
    def min_width(self):
        return self.image.shape[1]

    # Added in 0.4.0.
    def draw(self, height, width):
        image_rsp, size_rs, paddings = _resizepad_to_size(
            self.image, (height, width), cval=_COLOR_GRID_BACKGROUND)

        cbasoi = self.cbasoi.deepcopy()
        cbasoi = cbasoi.on_(size_rs)
        cbasoi = cbasoi.shift_(y=paddings[0], x=paddings[3])
        cbasoi.shape = image_rsp.shape

        return cbasoi.draw_on_image(image_rsp)


class _DebugGridColumn(object):
    """A single column within the debug image grid.

    Added in 0.4.0.

    """

    def __init__(self, cells):
        self.cells = cells

    @property
    def nb_rows(self):
        """Number of rows in the column, i.e. examples in batch.

        Added in 0.4.0.

        """
        return len(self.cells)

    @property
    def max_cell_width(self):
        """Width in pixels of the widest cell in the column.

        Added in 0.4.0.

        """
        return max([cell.min_width for cell in self.cells])

    @property
    def max_cell_height(self):
        """Height in pixels of the tallest cell in the column.

        Added in 0.4.0.

        """
        return max([cell.min_height for cell in self.cells])

    def draw(self, heights):
        """Convert this column to an image array.

        Added in 0.4.0.

        """
        width = self.max_cell_width
        return np.vstack([cell.draw(height=height, width=width)
                          for cell, height
                          in zip(self.cells, heights)])


class _DebugGrid(object):
    """A debug image grid.

    Columns correspond to the input datatypes (e.g. images, bounding boxes).
    Rows correspond to the examples within a batch.

    Added in 0.4.0.

    """

    # Added in 0.4.0.
    def __init__(self, columns):
        assert len(columns) > 0
        self.columns = columns

    def draw(self):
        """Convert this grid to an image array.

        Added in 0.4.0.

        """
        nb_rows_by_col = [column.nb_rows for column in self.columns]
        assert len(set(nb_rows_by_col)) == 1
        rowwise_heights = np.zeros((self.columns[0].nb_rows,), dtype=np.int32)
        for column in self.columns:
            heights = [cell.min_height for cell in column.cells]
            rowwise_heights = np.maximum(rowwise_heights, heights)
        return np.hstack([column.draw(heights=rowwise_heights)
                          for column in self.columns])


# TODO image subtitles
# TODO run start date
# TODO main process id, process id
# TODO warning if map aspect ratio is different from image aspect ratio
# TODO error if non-image shapes differ from image shapes
def draw_debug_image(images, heatmaps=None, segmentation_maps=None,
                     keypoints=None, bounding_boxes=None, polygons=None,
                     line_strings=None):
    """Generate a debug image grid of a single batch and various datatypes.

    Added in 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    images : ndarray or list of ndarray
        Images in the batch. Must always be provided. Batches without images
        cannot be visualized.

    heatmaps : None or list of imgaug.augmentables.heatmaps.HeatmapsOnImage, optional
        Heatmaps on the provided images.

    segmentation_maps : None or list of imgaug.augmentables.segmaps.SegmentationMapsOnImage, optional
        Segmentation maps on the provided images.

    keypoints : None or list of imgaug.augmentables.kps.KeypointsOnImage, optional
        Keypoints on the provided images.

    bounding_boxes : None or list of imgaug.augmentables.bbs.BoundingBoxesOnImage, optional
        Bounding boxes on the provided images.

    polygons : None or list of imgaug.augmentables.polys.PolygonsOnImage, optional
        Polygons on the provided images.

    line_strings : None or list of imgaug.augmentables.lines.LineStringsOnImage, optional
        Line strings on the provided images.

    Returns
    -------
    ndarray
        Visualized batch as RGB image.

    Examples
    --------
    >>> import numpy as np
    >>> import imgaug.augmenters as iaa
    >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
    >>> debug_image = iaa.draw_debug_image(images=[image, image])

    Generate a debug image for two empty images.

    >>> from imgaug.augmentables.kps import KeypointsOnImage
    >>> kpsoi = KeypointsOnImage.from_xy_array([(10.5, 20.5), (30.5, 30.5)],
    >>>                                        shape=image.shape)
    >>> debug_image = iaa.draw_debug_image(images=[image, image],
    >>>                                    keypoints=[kpsoi, kpsoi])

    Generate a debug image for two empty images, each having two keypoints
    drawn on them.

    >>> from imgaug.augmentables.batches import UnnormalizedBatch
    >>> segmap_arr = np.zeros((32, 32, 1), dtype=np.int32)
    >>> kp_tuples = [(10.5, 20.5), (30.5, 30.5)]
    >>> batch = UnnormalizedBatch(images=[image, image],
    >>>                           segmentation_maps=[segmap_arr, segmap_arr],
    >>>                           keypoints=[kp_tuples, kp_tuples])
    >>> batch = batch.to_normalized_batch()
    >>> debug_image = iaa.draw_debug_image(
    >>>     images=batch.images_unaug,
    >>>     segmentation_maps=batch.segmentation_maps_unaug,
    >>>     keypoints=batch.keypoints_unaug)

    Generate a debug image for two empty images, each having an empty
    segmentation map and two keypoints drawn on them. This example uses
    ``UnnormalizedBatch`` to show how to mostly evade going through imgaug
    classes.

    """
    columns = [_create_images_column(images)]

    if heatmaps is not None:
        columns.extend(_create_heatmaps_columns(heatmaps, images))

    if segmentation_maps is not None:
        columns.extend(_create_segmap_columns(segmentation_maps, images))

    if keypoints is not None:
        columns.append(_create_cbasois_column(keypoints, images, "Keypoints"))

    if bounding_boxes is not None:
        columns.append(_create_cbasois_column(bounding_boxes, images,
                                              "Bounding Boxes"))

    if polygons is not None:
        columns.append(_create_cbasois_column(polygons, images, "Polygons"))

    if line_strings is not None:
        columns.append(_create_cbasois_column(line_strings, images,
                                              "Line Strings"))

    result = _DebugGrid(columns)
    result = result.draw()
    result = sizelib.pad(result, top=1, right=1, bottom=1, left=1,
                         mode="constant", cval=_COLOR_GRID_BACKGROUND)
    return result


# Added in 0.4.0.
def _add_borders(cells):
    """Add a border (cell) around a cell."""
    return [_DebugGridBorderCell(1, _COLOR_GRID_BACKGROUND, cell)
            for cell in cells]


# Added in 0.4.0.
def _add_text_cell(title, cells):
    """Add a text cell before other cells."""
    return [_DebugGridTextCell(title)] + cells


# Added in 0.4.0.
def _create_images_column(images):
    """Create columns for image data."""
    cells = [_DebugGridImageCell(image) for image in images]
    images_descr = _generate_images_description(images)
    column = _DebugGridColumn(
        _add_borders(
            _add_text_cell(
                "Images",
                _add_text_cell(
                    images_descr,
                    cells)
            )
        )
    )
    return column


# Added in 0.4.0.
def _create_heatmaps_columns(heatmaps, images):
    """Create columns for heatmap data."""
    nb_map_channels = max([heatmap.arr_0to1.shape[2]
                           for heatmap in heatmaps])
    columns = [[] for _ in np.arange(nb_map_channels)]
    for image, heatmap in zip(images, heatmaps):
        heatmap_drawn = heatmap.draw()
        for c, heatmap_drawn_c in enumerate(heatmap_drawn):
            columns[c].append(
                _DebugGridImageCell(image, overlay=heatmap_drawn_c))

    columns = [
        _DebugGridColumn(
            _add_borders(
                _add_text_cell(
                    "Heatmaps",
                    _add_text_cell(
                        _generate_heatmaps_description(
                            heatmaps,
                            channel_idx=c,
                            show_details=(c == 0)),
                        cells)
                )
            )
        )
        for c, cells in enumerate(columns)
    ]
    return columns


# Added in 0.4.0.
def _create_segmap_columns(segmentation_maps, images):
    """Create columns for segmentation map data."""
    nb_map_channels = max([segmap.arr.shape[2]
                           for segmap in segmentation_maps])
    columns = [[] for _ in np.arange(nb_map_channels)]
    for image, segmap in zip(images, segmentation_maps):
        # TODO this currently draws the background in black, hence the
        #      resulting blended image is dark at class id 0
        segmap_drawn = segmap.draw()
        for c, segmap_drawn_c in enumerate(segmap_drawn):
            columns[c].append(
                _DebugGridImageCell(image, overlay=segmap_drawn_c))

    columns = [
        _DebugGridColumn(
            _add_borders(
                _add_text_cell(
                    "SegMaps",
                    _add_text_cell(
                        _generate_segmaps_description(
                            segmentation_maps,
                            channel_idx=c,
                            show_details=(c == 0)),
                        cells
                    )
                )
            )
        )
        for c, cells in enumerate(columns)
    ]

    return columns


# Added in 0.4.0.
def _create_cbasois_column(cbasois, images, column_name):
    """Create a column for coordinate-based augmentables."""
    cells = [_DebugGridCBAsOICell(cbasoi, image)
             for cbasoi, image
             in zip(cbasois, images)]
    descr = _generate_cbasois_description(cbasois, images)
    column = _DebugGridColumn(
        _add_borders(
            _add_text_cell(
                column_name,
                _add_text_cell(descr, cells)
            )
        )
    )
    return column


# Added in 0.4.0.
def _generate_images_description(images):
    """Generate description for image columns."""
    if ia.is_np_array(images):
        shapes_str = "array, shape %11s" % (str(images.shape),)
        dtypes_str = "dtype %8s" % (images.dtype.name,)
        if len(images) == 0:
            value_range_str = ""
        elif images.dtype.kind in ["u", "i", "b"]:
            value_range_str = "value range: %3d to %3d" % (
                np.min(images), np.max(images))
        else:
            value_range_str = "value range: %7.4f to %7.4f" % (
                np.min(images), np.max(images))
    else:
        stats = _ListOfArraysStats(images)

        if stats.empty:
            shapes_str = ""
        elif stats.all_same_shape:
            shapes_str = (
                "list of %3d arrays\n"
                "all shape %11s"
            ) % (len(images), stats.shapes[0],)
        else:
            shapes_str = (
                "list of %3d arrays\n"
                "varying shapes\n"
                "smallest image: %11s\n"
                "largest image: %11s\n"
                "height: %3d to %3d\n"
                "width: %3d to %3d\n"
                "channels: %1s to %1s"
            ) % (len(images),
                 stats.smallest_shape, stats.largest_shape,
                 stats.height_min, stats.height_max,
                 stats.width_min, stats.width_max,
                 stats.get_channels_min("None"),
                 stats.get_channels_max("None"))

        if stats.empty:
            dtypes_str = ""
        elif stats.all_same_dtype:
            dtypes_str = "all dtype %8s" % (stats.dtypes[0],)
        else:
            dtypes_str = "dtypes: %s" % (", ".join(stats.unique_dtype_names),)

        if stats.empty:
            value_range_str = ""
        else:
            value_range_str = "value range: %3d to %3d"
            if not stats.all_dtypes_intlike:
                value_range_str = "value range: %6.4f to %6.4f"
            value_range_str = value_range_str % (stats.value_min,
                                                 stats.value_max)

    strs = [shapes_str, dtypes_str, value_range_str]
    return _join_description_strs(strs)


# Added in 0.4.0.
def _generate_segmaps_description(segmaps, channel_idx, show_details):
    """Generate description for segmap columns."""
    if len(segmaps) == 0:
        return "empty list"

    strs = _generate_sm_hm_description(segmaps, channel_idx, show_details)

    arrs_channel = [segmap.arr[:, :, channel_idx] for segmap in segmaps]
    stats_channel = _ListOfArraysStats(arrs_channel)
    value_range_str = (
        "value range: %3d to %3d\n"
        "number of unique classes: %2d"
    ) % (stats_channel.value_min, stats_channel.value_max,
         stats_channel.nb_unique_values)

    return _join_description_strs(strs + [value_range_str])


# Added in 0.4.0.
def _generate_heatmaps_description(heatmaps, channel_idx, show_details):
    """Generate description for heatmap columns."""
    if len(heatmaps) == 0:
        return "empty list"

    strs = _generate_sm_hm_description(heatmaps, channel_idx, show_details)

    arrs_channel = [heatmap.arr_0to1[:, :, channel_idx] for heatmap in heatmaps]
    stats_channel = _ListOfArraysStats(arrs_channel)
    value_range_str = (
        "value range: %6.4f to %6.4f\n"
        "    (internal, max is [0.0, 1.0])"
    ) % (stats_channel.value_min, stats_channel.value_max)

    return _join_description_strs(strs + [value_range_str])


# Added in 0.4.0.
def _generate_sm_hm_description(augmentables, channel_idx, show_details):
    """Generate description for SegMap/Heatmap columns."""
    if augmentables is None:
        return ""
    if len(augmentables) == 0:
        return "empty list"

    arrs = [augmentable.get_arr() for augmentable in augmentables]
    stats = _ListOfArraysStats(arrs)

    if stats.get_channels_max(-1) > -1:
        channel_str = "Channel %1d of %1d" % (channel_idx+1,
                                              stats.get_channels_max(-1))
    else:
        channel_str = ""

    if not show_details:
        shapes_str = ""
    elif stats.all_same_shape:
        shapes_str = (
            "items for %3d images\n"
            "all arrays of shape %11s"
        ) % (len(augmentables), stats.shapes[0],)
    else:
        shapes_str = (
            "items for %3d images\n"
            "varying array shapes\n"
            "smallest: %11s\n"
            "largest: %11s\n"
            "height: %3d to %3d\n"
            "width: %3d to %3d\n"
            "channels: %1s to %1s"
        ) % (len(augmentables),
             stats.smallest_shape, stats.largest_shape,
             stats.height_min, stats.height_max,
             stats.width_min, stats.width_max,
             stats.get_channels_min("None"),
             stats.get_channels_max("None"))

    if not show_details:
        on_shapes_str = ""
    else:
        on_shapes_str = _generate_on_image_shapes_descr(augmentables)

    return [channel_str, shapes_str, on_shapes_str]


# Added in 0.4.0.
def _generate_cbasois_description(cbasois, images):
    """Generate description for coordinate-based augmentable columns."""
    images_str = "items for %d images" % (len(cbasois),)

    nb_items_lst = [len(cbasoi.items) for cbasoi in cbasois]
    nb_items_lst = nb_items_lst if len(cbasois) > 0 else [-1]
    nb_items = sum(nb_items_lst)
    items_str = (
        "fewest items on image: %3d\n"
        "most items on image: %3d\n"
        "total items: %6d"
    ) % (min(nb_items_lst), max(nb_items_lst), nb_items)

    areas = [
        cba.area if hasattr(cba, "area") else -1
        for cbasoi in cbasois
        for cba in cbasoi.items]
    areas = areas if len(cbasois) > 0 else [-1]
    areas_str = (
        "smallest area: %7.4f\n"
        "largest area: %7.4f"
    ) % (min(areas), max(areas))

    labels = list(ia.flatten([item.label if hasattr(item, "label") else None
                              for cbasoi in cbasois
                              for item in cbasoi.items]))
    labels_ctr = collections.Counter(labels)
    labels_most_common = []
    for label, count in labels_ctr.most_common(10):
        labels_most_common.append("\n    - %s (%3d, %6.2f%%)" % (
            label, count, count/nb_items * 100))
    labels_str = (
        "unique labels: %2d\n"
        "most common labels:"
        "%s"
    ) % (len(labels_ctr.keys()), "".join(labels_most_common))

    coords_ooi = []
    dists = []
    for cbasoi, image in zip(cbasois, images):
        h, w = image.shape[0:2]
        for cba in cbasoi.items:
            coords = cba.coords
            for coord in coords:
                x, y = coord
                dist = (x - w/2)**2 + (y - h/2) ** 2
                coords_ooi.append(not (0 <= x < w and 0 <= y < h))
                dists.append(((x, y), dist))

    # use x_ and y_ because otherwise we get a 'redefines x' error in pylint
    coords_extreme = [(x_, y_)
                      for (x_, y_), _
                      in sorted(dists, key=lambda t: t[1])]

    nb_ooi = sum(coords_ooi)
    ooi_str = (
        "coords out of image: %d (%6.2f%%)\n"
        "most extreme coord: (%5.1f, %5.1f)"
        # TODO "items anyhow out of image: %d (%.2f%%)\n"
        # TODO "items fully out of image: %d (%.2f%%)\n"
    ) % (nb_ooi, nb_ooi / len(coords_ooi) * 100,
         coords_extreme[-1][0], coords_extreme[-1][1])

    on_shapes_str = _generate_on_image_shapes_descr(cbasois)

    return _join_description_strs([images_str, items_str, areas_str,
                                   labels_str, ooi_str, on_shapes_str])


# Added in 0.4.0.
def _generate_on_image_shapes_descr(augmentables):
    """Generate text block for non-image data describing their image shapes."""
    on_shapes = [augmentable.shape for augmentable in augmentables]
    stats_imgs = _ListOfArraysStats([np.empty(on_shape)
                                     for on_shape in on_shapes])
    if stats_imgs.all_same_shape:
        on_shapes_str = "all on image shape %11s" % (stats_imgs.shapes[0],)
    else:
        on_shapes_str = (
            "on varying image shapes\n"
            "smallest image: %11s\n"
            "largest image: %11s"
        ) % (stats_imgs.smallest_shape, stats_imgs.largest_shape)
    return on_shapes_str


# Added in 0.4.0.
def _join_description_strs(strs):
    """Join lines to a single string while removing empty lines."""
    strs = [str_i for str_i in strs if len(str_i) > 0]
    return "\n".join(strs)


class _ListOfArraysStats(object):
    """Class to derive aggregated values from a list of arrays.

    E.g. shape of the largest array, number of unique dtypes etc.

    Added in 0.4.0.

    """

    def __init__(self, arrays):
        self.arrays = arrays

    # Added in 0.4.0.
    @property
    def empty(self):
        return len(self.arrays) == 0

    # Added in 0.4.0.
    @property
    def areas(self):
        return [np.prod(arr.shape[0:2]) for arr in self.arrays]

    # Added in 0.4.0.
    @property
    def arrays_by_area(self):
        arrays_by_area = [
            arr for arr, _
            in sorted(zip(self.arrays, self.areas), key=lambda t: t[1])
        ]
        return arrays_by_area

    # Added in 0.4.0.
    @property
    def shapes(self):
        return [arr.shape for arr in self.arrays]

    # Added in 0.4.0.
    @property
    def all_same_shape(self):
        if self.empty:
            return True
        return len(set(self.shapes)) == 1

    # Added in 0.4.0.
    @property
    def smallest_shape(self):
        if self.empty:
            return tuple()
        return self.arrays_by_area[0].shape

    # Added in 0.4.0.
    @property
    def largest_shape(self):
        if self.empty:
            return tuple()
        return self.arrays_by_area[-1].shape

    # Added in 0.4.0.
    @property
    def area_max(self):
        if self.empty:
            return tuple()
        return np.prod(self.arrays_by_area[-1][0:2])

    # Added in 0.4.0.
    @property
    def heights(self):
        return [arr.shape[0] for arr in self.arrays]

    # Added in 0.4.0.
    @property
    def height_min(self):
        heights = self.heights
        return min(heights) if len(heights) > 0 else 0

    # Added in 0.4.0.
    @property
    def height_max(self):
        heights = self.heights
        return max(heights) if len(heights) > 0 else 0

    # Added in 0.4.0.
    @property
    def widths(self):
        return [arr.shape[1] for arr in self.arrays]

    # Added in 0.4.0.
    @property
    def width_min(self):
        widths = self.widths
        return min(widths) if len(widths) > 0 else 0

    # Added in 0.4.0.
    @property
    def width_max(self):
        widths = self.widths
        return max(widths) if len(widths) > 0 else 0

    # Added in 0.4.0.
    def get_channels_min(self, default):
        if self.empty:
            return -1
        if any([arr.ndim == 2 for arr in self.arrays]):
            return default
        return min([arr.shape[2] for arr in self.arrays if arr.ndim > 2])

    # Added in 0.4.0.
    def get_channels_max(self, default):
        if self.empty:
            return -1
        if not any([arr.ndim > 2 for arr in self.arrays]):
            return default
        return max([arr.shape[2] for arr in self.arrays if arr.ndim > 2])

    # Added in 0.4.0.
    @property
    def dtypes(self):
        return [arr.dtype for arr in self.arrays]

    # Added in 0.4.0.
    @property
    def dtype_names(self):
        return [dtype.name for dtype in self.dtypes]

    # Added in 0.4.0.
    @property
    def all_same_dtype(self):
        return len(set(self.dtype_names)) in [0, 1]

    # Added in 0.4.0.
    @property
    def all_dtypes_intlike(self):
        if self.empty:
            return True
        return all([arr.dtype.kind in ["u", "i", "b"] for arr in self.arrays])

    # Added in 0.4.0.
    @property
    def unique_dtype_names(self):
        return sorted(list({arr.dtype.name for arr in self.arrays}))

    # Added in 0.4.0.
    @property
    def value_min(self):
        return min([np.min(arr) for arr in self.arrays])

    # Added in 0.4.0.
    @property
    def value_max(self):
        return max([np.max(arr) for arr in self.arrays])

    # Added in 0.4.0.
    @property
    def nb_unique_values(self):
        values_uq = set()
        for arr in self.arrays:
            values_uq.update(np.unique(arr))
        return len(values_uq)


# Added in 0.4.0.
@six.add_metaclass(ABCMeta)
class _IImageDestination(object):
    """A destination which receives images to save."""

    def on_batch(self, batch):
        """Signal to the destination that a new batch is processed.

        This is intended to be used by the destination e.g. to count batches.

        Added in 0.4.0.

        Parameters
        ----------
        batch : imgaug.augmentables.batches._BatchInAugmentation
            A batch to which the next ``receive()`` call may correspond.

        """

    def receive(self, image):
        """Receive and handle an image.

        Added in 0.4.0.

        Parameters
        ----------
        image : ndarray
            Image to be handled by the destination.

        """


# Added in 0.4.0.
class _MultiDestination(_IImageDestination):
    """A list of multiple destinations behaving like a single one."""

    # Added in 0.4.0.
    def __init__(self, destinations):
        self.destinations = destinations

    # Added in 0.4.0.
    def on_batch(self, batch):
        for destination in self.destinations:
            destination.on_batch(batch)

    # Added in 0.4.0.
    def receive(self, image):
        for destination in self.destinations:
            destination.receive(image)


# Added in 0.4.0.
class _FolderImageDestination(_IImageDestination):
    """A destination which saves images to a directory."""

    # Added in 0.4.0.
    def __init__(self, folder_path,
                 filename_pattern="batch_{batch_id:06d}.png"):
        super(_FolderImageDestination, self).__init__()
        self.folder_path = folder_path
        self.filename_pattern = filename_pattern
        self._batch_id = -1
        self._filepath = None

    # Added in 0.4.0.
    def on_batch(self, batch):
        self._batch_id += 1
        self._filepath = os.path.join(
            self.folder_path,
            self.filename_pattern.format(batch_id=self._batch_id))

    # Added in 0.4.0.
    def receive(self, image):
        imageio.imwrite(self._filepath, image)


# Added in 0.4.0.
@six.add_metaclass(ABCMeta)
class _IBatchwiseSchedule(object):
    """A schedule determining per batch whether a condition is met."""

    def on_batch(self, batch):
        """Determine for the given batch whether the condition is met.

        Added in 0.4.0.

        Parameters
        ----------
        batch : _BatchInAugmentation
            Batch for which to evaluate the condition.

        Returns
        -------
        bool
            Signal whether the condition is met.

        """


# Added in 0.4.0.
class _EveryNBatchesSchedule(_IBatchwiseSchedule):
    """A schedule that generates a signal at every ``N`` th batch.

    This schedule must be called for *every* batch in order to count them.

    Added in 0.4.0.

    """

    def __init__(self, interval):
        self.interval = interval
        self._batch_id = -1

    # Added in 0.4.0.
    def on_batch(self, batch):
        self._batch_id += 1
        signal = (self._batch_id % self.interval == 0)
        return signal


class _SaveDebugImage(meta.Augmenter):
    """Augmenter saving debug images to a destination according to a schedule.

    Added in 0.4.0.

    Parameters
    ----------
    destination : _IImageDestination
        The destination receiving debug images.

    schedule : _IBatchwiseSchedule
        The schedule to use to determine for which batches an image is
        supposed to be generated.

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

    # Added in 0.4.0.
    def __init__(self, destination, schedule,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(_SaveDebugImage, self).__init__(
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)
        self.destination = destination
        self.schedule = schedule

    # Added in 0.4.0.
    def _augment_batch_(self, batch, random_state, parents, hooks):
        save = self.schedule.on_batch(batch)
        self.destination.on_batch(batch)

        if save:
            image = draw_debug_image(
                images=batch.images,
                heatmaps=batch.heatmaps,
                segmentation_maps=batch.segmentation_maps,
                keypoints=batch.keypoints,
                bounding_boxes=batch.bounding_boxes,
                polygons=batch.polygons,
                line_strings=batch.line_strings)

            self.destination.receive(image)

        return batch


class SaveDebugImageEveryNBatches(_SaveDebugImage):
    """Visualize data in batches and save corresponding plots to a folder.

    Added in 0.4.0.

    **Supported dtypes**:

    See :func:`~imgaug.augmenters.debug.draw_debug_image`.

    Parameters
    ----------
    destination : str or _IImageDestination
        Path to a folder. The saved images will follow a filename pattern
        of ``batch_<batch_id>.png``. The latest image will additionally be
        saved to ``latest.png``.

    interval : int
        Interval in batches. If set to ``N``, every ``N`` th batch an
        image will be generated and saved, starting with the first observed
        batch.
        Note that the augmenter only counts batches that it sees. If it is
        executed conditionally or re-instantiated, it may not see all batches
        or the counter may be wrong in other ways.

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
    >>> import tempfile
    >>> folder_path = tempfile.mkdtemp()
    >>> seq = iaa.Sequential([
    >>>     iaa.Sequential([
    >>>         iaa.Fliplr(0.5),
    >>>         iaa.Crop(px=(0, 16))
    >>>     ], random_order=True),
    >>>     iaa.SaveDebugImageEveryNBatches(folder_path, 100)
    >>> ])

    """

    # Added in 0.4.0.
    def __init__(self, destination, interval,
                 seed=None, name=None,
                 random_state="deprecated", deterministic="deprecated"):
        schedule = _EveryNBatchesSchedule(interval)
        if not isinstance(destination, _IImageDestination):
            assert os.path.isdir(destination), (
                "Expected 'destination' to be a string path to an existing "
                "directory. Got path '%s'." % (destination,))
            destination = _MultiDestination([
                _FolderImageDestination(destination),
                _FolderImageDestination(destination,
                                        filename_pattern="batch_latest.png")
            ])
        super(SaveDebugImageEveryNBatches, self).__init__(
            destination=destination, schedule=schedule,
            seed=seed, name=name,
            random_state=random_state, deterministic=deterministic)

    # Added in 0.4.0.
    def get_parameters(self):
        dests = self.destination.destinations
        return [
            dests[0].folder_path,
            dests[0].filename_pattern,
            dests[1].folder_path,
            dests[1].filename_pattern,
            self.schedule.interval
        ]
