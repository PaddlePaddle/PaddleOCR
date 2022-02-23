"""Classes dealing with segmentation maps.

E.g. masks, semantic or instance segmentation maps.

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm

from .. import imgaug as ia
from ..augmenters import blend as blendlib
from .base import IAugmentable


@ia.deprecated(alt_func="SegmentationMapsOnImage",
               comment="(Note the plural 'Maps' instead of old 'Map'.)")
def SegmentationMapOnImage(*args, **kwargs):
    """Object representing a segmentation map associated with an image."""
    # pylint: disable=invalid-name
    return SegmentationMapsOnImage(*args, **kwargs)


class SegmentationMapsOnImage(IAugmentable):
    """
    Object representing a segmentation map associated with an image.

    Attributes
    ----------
    DEFAULT_SEGMENT_COLORS : list of tuple of int
        Standard RGB colors to use during drawing, ordered by class index.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Array representing the segmentation map(s). May have dtypes bool,
        int or uint.

    shape : tuple of int
        Shape of the image on which the segmentation map(s) is/are placed.
        **Not** the shape of the segmentation map(s) array, unless it is
        identical to the image shape (note the likely difference between the
        arrays in the number of channels).
        This is expected to be ``(H, W)`` or ``(H, W, C)`` with ``C`` usually
        being ``3``.
        If there is no corresponding image, use ``(H_arr, W_arr)`` instead,
        where ``H_arr`` is the height of the segmentation map(s) array
        (analogous ``W_arr``).

    nb_classes : None or int, optional
        Deprecated.

    """

    # TODO replace this by matplotlib colormap
    DEFAULT_SEGMENT_COLORS = [
        (0, 0, 0),  # black
        (230, 25, 75),  # red
        (60, 180, 75),  # green
        (255, 225, 25),  # yellow
        (0, 130, 200),  # blue
        (245, 130, 48),  # orange
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190),  # pink
        (0, 128, 128),  # teal
        (230, 190, 255),  # lavender
        (170, 110, 40),  # brown
        (255, 250, 200),  # beige
        (128, 0, 0),  # maroon
        (170, 255, 195),  # mint
        (128, 128, 0),  # olive
        (255, 215, 180),  # coral
        (0, 0, 128),  # navy
        (128, 128, 128),  # grey
        (255, 255, 255),  # white
        # --
        (115, 12, 37),  # dark red
        (30, 90, 37),  # dark green
        (127, 112, 12),  # dark yellow
        (0, 65, 100),  # dark blue
        (122, 65, 24),  # dark orange
        (72, 15, 90),  # dark purple
        (35, 120, 120),  # dark cyan
        (120, 25, 115),  # dark magenta
        (105, 122, 30),  # dark lime
        (125, 95, 95),  # dark pink
        (0, 64, 64),  # dark teal
        (115, 95, 127),  # dark lavender
        (85, 55, 20),  # dark brown
        (127, 125, 100),  # dark beige
        (64, 0, 0),  # dark maroon
        (85, 127, 97),  # dark mint
        (64, 64, 0),  # dark olive
        (127, 107, 90),  # dark coral
        (0, 0, 64),  # dark navy
        (64, 64, 64),  # dark grey
    ]

    def __init__(self, arr, shape, nb_classes=None):
        assert ia.is_np_array(arr), (
            "Expected to get numpy array, got %s." % (type(arr),))
        assert arr.ndim in [2, 3], (
            "Expected segmentation map array to be 2- or "
            "3-dimensional, got %d dimensions and shape %s." % (
                arr.ndim, arr.shape))
        assert isinstance(shape, tuple), (
            "Expected 'shape' to be a tuple denoting the shape of the image "
            "on which the segmentation map is placed. Got type %s instead." % (
                type(shape)))

        if arr.dtype.kind == "f":
            ia.warn_deprecated(
                "Got a float array as the segmentation map in "
                "SegmentationMapsOnImage. That is deprecated. Please provide "
                "instead a (H,W,[C]) array of dtype bool_, int or uint, where "
                "C denotes the segmentation map index."
            )

            if arr.ndim == 2:
                arr = (arr > 0.5)
            else:  # arr.ndim == 3
                arr = np.argmax(arr, axis=2).astype(np.int32)

        if arr.dtype.name == "bool":
            self._input_was = (arr.dtype, arr.ndim)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
        elif arr.dtype.kind in ["i", "u"]:
            assert np.min(arr.flat[0:100]) >= 0, (
                "Expected segmentation map array to only contain values >=0, "
                "got a minimum of %d." % (np.min(arr),))
            if arr.dtype.kind == "u":
                # allow only <=uint16 due to conversion to int32
                assert arr.dtype.itemsize <= 2, (
                    "When using uint arrays as segmentation maps, only uint8 "
                    "and uint16 are allowed. Got dtype %s." % (arr.dtype.name,)
                )
            elif arr.dtype.kind == "i":
                # allow only <=uint16 due to conversion to int32
                assert arr.dtype.itemsize <= 4, (
                    "When using int arrays as segmentation maps, only int8, "
                    "int16 and int32 are allowed. Got dtype %s." % (
                        arr.dtype.name,)
                )

            self._input_was = (arr.dtype, arr.ndim)
            if arr.ndim == 2:
                arr = arr[..., np.newaxis]
        else:
            raise Exception((
                "Input was expected to be an array of dtype 'bool', 'int' "
                "or 'uint'. Got dtype '%s'.") % (arr.dtype.name,))

        if arr.dtype.name != "int32":
            arr = arr.astype(np.int32)

        self.arr = arr

        # don't allow arrays here as an alternative to tuples as input
        # as allowing arrays introduces risk to mix up 'arr' and 'shape' args
        self.shape = shape

        if nb_classes is not None:
            ia.warn_deprecated(
                "Providing nb_classes to SegmentationMapsOnImage is no longer "
                "necessary and hence deprecated. The argument is ignored "
                "and can be safely removed.")

    def get_arr(self):
        """Return the seg.map array, with original dtype and shape ndim.

        Here, "original" denotes the dtype and number of shape dimensions that
        was used when the :class:`SegmentationMapsOnImage` instance was
        created, i.e. upon the call of
        :func:`SegmentationMapsOnImage.__init__`.
        Internally, this class may use a different dtype and shape to simplify
        computations.

        .. note::

            The height and width may have changed compared to the original
            input due to e.g. pooling operations.

        Returns
        -------
        ndarray
            Segmentation map array.
            Same dtype and number of dimensions as was originally used when
            the :class:`SegmentationMapsOnImage` instance was created.

        """
        input_dtype, input_ndim = self._input_was
        # The internally used int32 has a wider value range than any other
        # input dtype, hence we can simply convert via astype() here.
        arr_input = self.arr.astype(input_dtype)
        if input_ndim == 2:
            assert arr_input.shape[2] == 1, (
                "Originally got a (H,W) segmentation map. Internal array "
                "should now have shape (H,W,1), but got %s. This might be "
                "an internal error." % (arr_input.shape,))
            return arr_input[:, :, 0]
        return arr_input

    @ia.deprecated(alt_func="SegmentationMapsOnImage.get_arr()")
    def get_arr_int(self, *args, **kwargs):
        """Return the seg.map array, with original dtype and shape ndim."""
        # pylint: disable=unused-argument
        return self.get_arr()

    def draw(self, size=None, colors=None):
        """
        Render the segmentation map as an RGB image.

        Parameters
        ----------
        size : None or float or iterable of int or iterable of float, optional
            Size of the rendered RGB image as ``(height, width)``.
            See :func:`~imgaug.imgaug.imresize_single_image` for details.
            If set to ``None``, no resizing is performed and the size of the
            segmentation map array is used.

        colors : None or list of tuple of int, optional
            Colors to use. One for each class to draw.
            If ``None``, then default colors will be used.

        Returns
        -------
        list of (H,W,3) ndarray
            Rendered segmentation map (dtype is ``uint8``).
            One per ``C`` in the original input array ``(H,W,C)``.

        """
        def _handle_sizeval(sizeval, arr_axis_size):
            if sizeval is None:
                return arr_axis_size
            if ia.is_single_float(sizeval):
                return max(int(arr_axis_size * sizeval), 1)
            if ia.is_single_integer(sizeval):
                return sizeval
            raise ValueError("Expected float or int, got %s." % (
                type(sizeval),))

        if size is None:
            size = [size, size]
        elif not ia.is_iterable(size):
            size = [size, size]

        height = _handle_sizeval(size[0], self.arr.shape[0])
        width = _handle_sizeval(size[1], self.arr.shape[1])
        image = np.zeros((height, width, 3), dtype=np.uint8)

        return self.draw_on_image(
            image,
            alpha=1.0,
            resize="segmentation_map",
            colors=colors,
            draw_background=True
        )

    def draw_on_image(self, image, alpha=0.75, resize="segmentation_map",
                      colors=None, draw_background=False,
                      background_class_id=0, background_threshold=None):
        """Draw the segmentation map as an overlay over an image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            Image onto which to draw the segmentation map. Expected dtype
            is ``uint8``.

        alpha : float, optional
            Alpha/opacity value to use for the mixing of image and
            segmentation map. Larger values mean that the segmentation map
            will be more visible and the image less visible.

        resize : {'segmentation_map', 'image'}, optional
            In case of size differences between the image and segmentation
            map, either the image or the segmentation map can be resized.
            This parameter controls which of the two will be resized to the
            other's size.

        colors : None or list of tuple of int, optional
            Colors to use. One for each class to draw.
            If ``None``, then default colors will be used.

        draw_background : bool, optional
            If ``True``, the background will be drawn like any other class.
            If ``False``, the background will not be drawn, i.e. the respective
            background pixels will be identical with the image's RGB color at
            the corresponding spatial location and no color overlay will be
            applied.

        background_class_id : int, optional
            Class id to interpret as the background class.
            See `draw_background`.

        background_threshold : None, optional
            Deprecated.
            This parameter is ignored.

        Returns
        -------
        list of (H,W,3) ndarray
            Rendered overlays as ``uint8`` arrays.
            Always a **list** containing one RGB image per segmentation map
            array channel.

        """
        if background_threshold is not None:
            ia.warn_deprecated(
                "The argument `background_threshold` is deprecated and "
                "ignored. Please don't use it anymore.")

        assert image.ndim == 3, (
            "Expected to draw on 3-dimensional image, got image with %d "
            "dimensions." % (image.ndim,))
        assert image.shape[2] == 3, (
            "Expected to draw on RGB image, got image with %d channels "
            "instead." % (image.shape[2],))
        assert image.dtype.name == "uint8", (
            "Expected to get image with dtype uint8, got dtype %s." % (
                image.dtype.name,))
        assert 0 - 1e-8 <= alpha <= 1.0 + 1e-8, (
            "Expected 'alpha' to be in interval [0.0, 1.0], got %.4f." % (
                alpha,))
        assert resize in ["segmentation_map", "image"], (
            "Expected 'resize' to be \"segmentation_map\" or \"image\", got "
            "%s." % (resize,))

        colors = (
            colors
            if colors is not None
            else SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS
        )

        if resize == "image":
            image = ia.imresize_single_image(
                image, self.arr.shape[0:2], interpolation="cubic")

        segmaps_drawn = []
        arr_channelwise = np.dsplit(self.arr, self.arr.shape[2])
        for arr in arr_channelwise:
            arr = arr[:, :, 0]

            nb_classes = 1 + np.max(arr)
            segmap_drawn = np.zeros((arr.shape[0], arr.shape[1], 3),
                                    dtype=np.uint8)
            assert nb_classes <= len(colors), (
                "Can't draw all %d classes as it would exceed the maximum "
                "number of %d available colors." % (nb_classes, len(colors),))

            ids_in_map = np.unique(arr)
            for c, color in zip(sm.xrange(nb_classes), colors):
                if c in ids_in_map:
                    class_mask = (arr == c)
                    segmap_drawn[class_mask] = color

            segmap_drawn = ia.imresize_single_image(
                segmap_drawn, image.shape[0:2], interpolation="nearest")

            segmap_on_image = blendlib.blend_alpha(segmap_drawn, image, alpha)

            if draw_background:
                mix = segmap_on_image
            else:
                foreground_mask = ia.imresize_single_image(
                    (arr != background_class_id),
                    image.shape[0:2],
                    interpolation="nearest")
                # without this, the merge below does nothing
                foreground_mask = np.atleast_3d(foreground_mask)

                mix = (
                    (~foreground_mask) * image
                    + foreground_mask * segmap_on_image
                )
            segmaps_drawn.append(mix)
        return segmaps_drawn

    def pad(self, top=0, right=0, bottom=0, left=0, mode="constant", cval=0):
        """Pad the segmentation maps at their top/right/bottom/left side.

        Parameters
        ----------
        top : int, optional
            Amount of pixels to add at the top side of the segmentation map.
            Must be ``0`` or greater.

        right : int, optional
            Amount of pixels to add at the right side of the segmentation map.
            Must be ``0`` or greater.

        bottom : int, optional
            Amount of pixels to add at the bottom side of the segmentation map.
            Must be ``0`` or greater.

        left : int, optional
            Amount of pixels to add at the left side of the segmentation map.
            Must be ``0`` or greater.

        mode : str, optional
            Padding mode to use. See :func:`~imgaug.imgaug.pad` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``.
            See :func:`~imgaug.imgaug.pad` for details.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Padded segmentation map with height ``H'=H+top+bottom`` and
            width ``W'=W+left+right``.

        """
        from ..augmenters import size as iasize
        arr_padded = iasize.pad(self.arr, top=top, right=right, bottom=bottom,
                                left=left, mode=mode, cval=cval)
        return self.deepcopy(arr=arr_padded)

    def pad_to_aspect_ratio(self, aspect_ratio, mode="constant", cval=0,
                            return_pad_amounts=False):
        """Pad the segmentation maps until they match a target aspect ratio.

        Depending on which dimension is smaller (height or width), only the
        corresponding sides (left/right or top/bottom) will be padded. In
        each case, both of the sides will be padded equally.

        Parameters
        ----------
        aspect_ratio : float
            Target aspect ratio, given as width/height. E.g. ``2.0`` denotes
            the image having twice as much width as height.

        mode : str, optional
            Padding mode to use.
            See :func:`~imgaug.imgaug.pad` for details.

        cval : number, optional
            Value to use for padding if `mode` is ``constant``.
            See :func:`~imgaug.imgaug.pad` for details.

        return_pad_amounts : bool, optional
            If ``False``, then only the padded instance will be returned.
            If ``True``, a tuple with two entries will be returned, where
            the first entry is the padded instance and the second entry are
            the amounts by which each array side was padded. These amounts are
            again a tuple of the form ``(top, right, bottom, left)``, with
            each value being an integer.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Padded segmentation map as :class:`SegmentationMapsOnImage`
            instance.

        tuple of int
            Amounts by which the instance's array was padded on each side,
            given as a tuple ``(top, right, bottom, left)``.
            This tuple is only returned if `return_pad_amounts` was set to
            ``True``.

        """
        from ..augmenters import size as iasize
        arr_padded, pad_amounts = iasize.pad_to_aspect_ratio(
            self.arr,
            aspect_ratio=aspect_ratio,
            mode=mode,
            cval=cval,
            return_pad_amounts=True)
        segmap = self.deepcopy(arr=arr_padded)
        if return_pad_amounts:
            return segmap, pad_amounts
        return segmap

    @ia.deprecated(alt_func="SegmentationMapsOnImage.resize()",
                   comment="resize() has the exactly same interface.")
    def scale(self, *args, **kwargs):
        """Resize the seg.map(s) array given a target size and interpolation."""
        return self.resize(*args, **kwargs)

    def resize(self, sizes, interpolation="nearest"):
        """Resize the seg.map(s) array given a target size and interpolation.

        Parameters
        ----------
        sizes : float or iterable of int or iterable of float
            New size of the array in ``(height, width)``.
            See :func:`~imgaug.imgaug.imresize_single_image` for details.

        interpolation : None or str or int, optional
            The interpolation to use during resize.
            Nearest neighbour interpolation (``"nearest"``) is almost always
            the best choice.
            See :func:`~imgaug.imgaug.imresize_single_image` for details.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Resized segmentation map object.

        """
        arr_resized = ia.imresize_single_image(self.arr, sizes,
                                               interpolation=interpolation)
        return self.deepcopy(arr_resized)

    # TODO how best to handle changes to _input_was due to changed 'arr'?
    def copy(self, arr=None, shape=None):
        """Create a shallow copy of the segmentation map object.

        Parameters
        ----------
        arr : None or (H,W) ndarray or (H,W,C) ndarray, optional
            Optionally the `arr` attribute to use for the new segmentation map
            instance. Will be copied from the old instance if not provided.
            See
            :func:`~imgaug.augmentables.segmaps.SegmentationMapsOnImage.__init__`
            for details.

        shape : None or tuple of int, optional
            Optionally the shape attribute to use for the the new segmentation
            map instance. Will be copied from the old instance if not provided.
            See
            :func:`~imgaug.augmentables.segmaps.SegmentationMapsOnImage.__init__`
            for details.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Shallow copy.

        """
        # pylint: disable=protected-access
        segmap = SegmentationMapsOnImage(
            self.arr if arr is None else arr,
            shape=self.shape if shape is None else shape)
        segmap._input_was = self._input_was
        return segmap

    def deepcopy(self, arr=None, shape=None):
        """Create a deep copy of the segmentation map object.

        Parameters
        ----------
        arr : None or (H,W) ndarray or (H,W,C) ndarray, optional
            Optionally the `arr` attribute to use for the new segmentation map
            instance. Will be copied from the old instance if not provided.
            See
            :func:`~imgaug.augmentables.segmaps.SegmentationMapsOnImage.__init__`
            for details.

        shape : None or tuple of int, optional
            Optionally the shape attribute to use for the the new segmentation
            map instance. Will be copied from the old instance if not provided.
            See
            :func:`~imgaug.augmentables.segmaps.SegmentationMapsOnImage.__init__`
            for details.

        Returns
        -------
        imgaug.augmentables.segmaps.SegmentationMapsOnImage
            Deep copy.

        """
        # pylint: disable=protected-access
        segmap = SegmentationMapsOnImage(
            np.copy(self.arr if arr is None else arr),
            shape=self.shape if shape is None else shape)
        segmap._input_was = self._input_was
        return segmap
