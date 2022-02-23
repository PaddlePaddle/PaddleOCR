"""Classes to represent heatmaps, i.e. float arrays of ``[0.0, 1.0]``."""
from __future__ import print_function, division, absolute_import

import numpy as np
import six.moves as sm

from .. import imgaug as ia
from .base import IAugmentable


class HeatmapsOnImage(IAugmentable):
    """Object representing heatmaps on a single image.

    Parameters
    ----------
    arr : (H,W) ndarray or (H,W,C) ndarray
        Array representing the heatmap(s) on a single image.
        Multiple heatmaps may be provided, in which case ``C`` is expected to
        denote the heatmap index.
        The array must be of dtype ``float32``.

    shape : tuple of int
        Shape of the image on which the heatmap(s) is/are placed.
        **Not** the shape of the heatmap(s) array, unless it is identical
        to the image shape (note the likely difference between the arrays
        in the number of channels).
        This is expected to be ``(H, W)`` or ``(H, W, C)`` with ``C`` usually
        being ``3``.
        If there is no corresponding image, use ``(H_arr, W_arr)`` instead,
        where ``H_arr`` is the height of the heatmap(s) array
        (analogous ``W_arr``).

    min_value : float, optional
        Minimum value for the heatmaps that `arr` represents. This will
        usually be ``0.0``.

    max_value : float, optional
        Maximum value for the heatmaps that `arr` represents. This will
        usually be ``1.0``.

    """

    def __init__(self, arr, shape, min_value=0.0, max_value=1.0):
        """Construct a new HeatmapsOnImage object."""
        assert ia.is_np_array(arr), (
            "Expected numpy array as heatmap input array, "
            "got type %s" % (type(arr),))
        # TODO maybe allow 0-sized heatmaps? in that case the min() and max()
        #      must be adjusted
        assert arr.shape[0] > 0 and arr.shape[1] > 0, (
            "Expected numpy array as heatmap with height and width greater "
            "than 0, got shape %s." % (arr.shape,))
        assert arr.dtype.name in ["float32"], (
            "Heatmap input array expected to be of dtype float32, "
            "got dtype %s." % (arr.dtype,))
        assert arr.ndim in [2, 3], (
            "Heatmap input array must be 2d or 3d, got shape %s." % (
                arr.shape,))
        assert len(shape) in [2, 3], (
            "Argument 'shape' in HeatmapsOnImage expected to be 2d or 3d, "
            "got shape %s." % (shape,))
        assert min_value < max_value, (
            "Expected min_value to be lower than max_value, "
            "got %.4f and %.4f" % (min_value, max_value))

        eps = np.finfo(arr.dtype).eps
        components = arr.flat[0:50]
        beyond_min = np.min(components) < min_value - eps
        beyond_max = np.max(components) > max_value + eps
        if beyond_min or beyond_max:
            ia.warn(
                "Value range of heatmap was chosen to be (%.8f, %.8f), but "
                "found actual min/max of (%.8f, %.8f). Array will be "
                "clipped to chosen value range." % (
                    min_value, max_value, np.min(arr), np.max(arr)))
            arr = np.clip(arr, min_value, max_value)

        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
            self.arr_was_2d = True
        else:
            self.arr_was_2d = False

        min_is_zero = 0.0 - eps < min_value < 0.0 + eps
        max_is_one = 1.0 - eps < max_value < 1.0 + eps
        if min_is_zero and max_is_one:
            self.arr_0to1 = arr
        else:
            self.arr_0to1 = (arr - min_value) / (max_value - min_value)

        # don't allow arrays here as an alternative to tuples as input
        # as allowing arrays introduces risk to mix up 'arr' and 'shape' args
        self.shape = shape

        self.min_value = min_value
        self.max_value = max_value

    def get_arr(self):
        """Get the heatmap's array in value range provided to ``__init__()``.

        The :class:`HeatmapsOnImage` object saves heatmaps internally in the
        value range ``[0.0, 1.0]``. This function converts the internal
        representation to ``[min, max]``, where ``min`` and ``max`` are
        provided to :func:`HeatmapsOnImage.__init__` upon instantiation of
        the object.

        Returns
        -------
        (H,W) ndarray or (H,W,C) ndarray
            Heatmap array of dtype ``float32``.

        """
        if self.arr_was_2d and self.arr_0to1.shape[2] == 1:
            arr = self.arr_0to1[:, :, 0]
        else:
            arr = self.arr_0to1

        eps = np.finfo(np.float32).eps
        min_is_zero = 0.0 - eps < self.min_value < 0.0 + eps
        max_is_one = 1.0 - eps < self.max_value < 1.0 + eps
        if min_is_zero and max_is_one:
            return np.copy(arr)

        diff = self.max_value - self.min_value
        return self.min_value + diff * arr

    # TODO
    # def find_global_maxima(self):
    #    raise NotImplementedError()

    def draw(self, size=None, cmap="jet"):
        """Render the heatmaps as RGB images.

        Parameters
        ----------
        size : None or float or iterable of int or iterable of float, optional
            Size of the rendered RGB image as ``(height, width)``.
            See :func:`~imgaug.imgaug.imresize_single_image` for details.
            If set to ``None``, no resizing is performed and the size of the
            heatmaps array is used.

        cmap : str or None, optional
            Name of the ``matplotlib`` color map to use when convert the
            heatmaps to RGB images.
            If set to ``None``, no color map will be used and the heatmaps
            will be converted to simple intensity maps.

        Returns
        -------
        list of (H,W,3) ndarray
            Rendered heatmaps as ``uint8`` arrays.
            Always a **list** containing one RGB image per heatmap array
            channel.

        """
        heatmaps_uint8 = self.to_uint8()
        heatmaps_drawn = []

        for c in sm.xrange(heatmaps_uint8.shape[2]):
            # We use c:c+1 here to get a (H,W,1) array. Otherwise imresize
            # would have to re-attach an axis.
            heatmap_c = heatmaps_uint8[..., c:c+1]

            if size is not None:
                heatmap_c_rs = ia.imresize_single_image(
                    heatmap_c, size, interpolation="nearest")
            else:
                heatmap_c_rs = heatmap_c
            heatmap_c_rs = np.squeeze(heatmap_c_rs).astype(np.float32) / 255.0

            if cmap is not None:
                # import only when necessary (faster startup; optional
                # dependency; less fragile -- see issue #225)
                import matplotlib.pyplot as plt

                cmap_func = plt.get_cmap(cmap)
                heatmap_cmapped = cmap_func(heatmap_c_rs)
                heatmap_cmapped = np.delete(heatmap_cmapped, 3, 2)
            else:
                heatmap_cmapped = np.tile(
                    heatmap_c_rs[..., np.newaxis], (1, 1, 3))

            heatmap_cmapped = np.clip(
                heatmap_cmapped * 255, 0, 255).astype(np.uint8)

            heatmaps_drawn.append(heatmap_cmapped)
        return heatmaps_drawn

    def draw_on_image(self, image, alpha=0.75, cmap="jet", resize="heatmaps"):
        """Draw the heatmaps as overlays over an image.

        Parameters
        ----------
        image : (H,W,3) ndarray
            Image onto which to draw the heatmaps.
            Expected to be of dtype ``uint8``.

        alpha : float, optional
            Alpha/opacity value to use for the mixing of image and heatmaps.
            Larger values mean that the heatmaps will be more visible and the
            image less visible.

        cmap : str or None, optional
            Name of the ``matplotlib`` color map to use.
            See :func:`HeatmapsOnImage.draw` for details.

        resize : {'heatmaps', 'image'}, optional
            In case of size differences between the image and heatmaps,
            either the image or the heatmaps can be resized. This parameter
            controls which of the two will be resized to the other's size.

        Returns
        -------
        list of (H,W,3) ndarray
            Rendered overlays as ``uint8`` arrays.
            Always a **list** containing one RGB image per heatmap array
            channel.

        """
        # assert RGB image
        assert image.ndim == 3, (
            "Expected to draw on three-dimensional image, "
            "got %d dimensions with shape %s instead." % (
                image.ndim, image.shape))
        assert image.shape[2] == 3, (
            "Expected RGB image, got %d channels instead." % (image.shape[2],))
        assert image.dtype.name == "uint8", (
            "Expected uint8 image, got dtype %s." % (image.dtype.name,))

        assert 0 - 1e-8 <= alpha <= 1.0 + 1e-8, (
            "Expected 'alpha' to be in the interval [0.0, 1.0], got %.4f" % (
                alpha))
        assert resize in ["heatmaps", "image"], (
            "Expected resize to be \"heatmaps\" or \"image\", "
            "got %s instead." % (resize,))

        if resize == "image":
            image = ia.imresize_single_image(
                image, self.arr_0to1.shape[0:2], interpolation="cubic")

        heatmaps_drawn = self.draw(
            size=image.shape[0:2] if resize == "heatmaps" else None,
            cmap=cmap)

        # TODO use blend_alpha here
        mix = [
            np.clip(
                (1-alpha) * image + alpha * heatmap_i,
                0, 255
            ).astype(np.uint8)
            for heatmap_i
            in heatmaps_drawn]

        return mix

    def invert(self):
        """Invert each component in the heatmap.

        This shifts low values towards high values and vice versa.

        This changes each value to::

            v' = max - (v - min)

        where ``v`` is the value at a spatial location, ``min`` is the
        minimum value in the heatmap and ``max`` is the maximum value.
        As the heatmap uses internally a ``0.0`` to ``1.0`` representation,
        this simply becomes ``v' = 1.0 - v``.

        This function can be useful e.g. when working with depth maps, where
        algorithms might have an easier time representing the furthest away
        points with zeros, requiring an inverted depth map.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Inverted heatmap.

        """
        arr_inv = HeatmapsOnImage.from_0to1(
            1 - self.arr_0to1,
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)
        arr_inv.arr_was_2d = self.arr_was_2d
        return arr_inv

    def pad(self, top=0, right=0, bottom=0, left=0, mode="constant", cval=0.0):
        """Pad the heatmaps at their top/right/bottom/left side.

        Parameters
        ----------
        top : int, optional
            Amount of pixels to add at the top side of the heatmaps.
            Must be ``0`` or greater.

        right : int, optional
            Amount of pixels to add at the right side of the heatmaps.
            Must be ``0`` or greater.

        bottom : int, optional
            Amount of pixels to add at the bottom side of the heatmaps.
            Must be ``0`` or greater.

        left : int, optional
            Amount of pixels to add at the left side of the heatmaps.
            Must be ``0`` or greater.

        mode : string, optional
            Padding mode to use. See :func:`~imgaug.imgaug.pad` for details.

        cval : number, optional
            Value to use for padding `mode` is ``constant``.
            See :func:`~imgaug.imgaug.pad` for details.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Padded heatmaps of height ``H'=H+top+bottom`` and
            width ``W'=W+left+right``.

        """
        from ..augmenters import size as iasize
        arr_0to1_padded = iasize.pad(
            self.arr_0to1,
            top=top,
            right=right,
            bottom=bottom,
            left=left,
            mode=mode,
            cval=cval)
        # TODO change to deepcopy()
        return HeatmapsOnImage.from_0to1(
            arr_0to1_padded,
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)

    def pad_to_aspect_ratio(self, aspect_ratio, mode="constant", cval=0.0,
                            return_pad_amounts=False):
        """Pad the heatmaps until they match a target aspect ratio.

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
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Padded heatmaps as :class:`HeatmapsOnImage` instance.

        tuple of int
            Amounts by which the instance's array was padded on each side,
            given as a tuple ``(top, right, bottom, left)``.
            This tuple is only returned if `return_pad_amounts` was set to
            ``True``.

        """
        from ..augmenters import size as iasize
        arr_0to1_padded, pad_amounts = iasize.pad_to_aspect_ratio(
            self.arr_0to1,
            aspect_ratio=aspect_ratio,
            mode=mode,
            cval=cval,
            return_pad_amounts=True)
        # TODO change to deepcopy()
        heatmaps = HeatmapsOnImage.from_0to1(
            arr_0to1_padded,
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)
        if return_pad_amounts:
            return heatmaps, pad_amounts
        return heatmaps

    def avg_pool(self, block_size):
        """Average-pool the heatmap(s) array using a given block/kernel size.

        Parameters
        ----------
        block_size : int or tuple of int
            Size of each block of values to pool, aka kernel size.
            See :func:`~imgaug.imgaug.pool` for details.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps after average pooling.

        """
        arr_0to1_reduced = ia.avg_pool(self.arr_0to1, block_size, pad_cval=0.0)
        return HeatmapsOnImage.from_0to1(
            arr_0to1_reduced,
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)

    def max_pool(self, block_size):
        """Max-pool the heatmap(s) array using a given block/kernel size.

        Parameters
        ----------
        block_size : int or tuple of int
            Size of each block of values to pool, aka kernel size.
            See :func:`~imgaug.imgaug.pool` for details.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps after max-pooling.

        """
        arr_0to1_reduced = ia.max_pool(self.arr_0to1, block_size)
        return HeatmapsOnImage.from_0to1(
            arr_0to1_reduced,
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)

    @ia.deprecated(alt_func="HeatmapsOnImage.resize()",
                   comment="resize() has the exactly same interface.")
    def scale(self, *args, **kwargs):
        """Resize the heatmap(s) array given a target size and interpolation."""
        return self.resize(*args, **kwargs)

    def resize(self, sizes, interpolation="cubic"):
        """Resize the heatmap(s) array given a target size and interpolation.

        Parameters
        ----------
        sizes : float or iterable of int or iterable of float
            New size of the array in ``(height, width)``.
            See :func:`~imgaug.imgaug.imresize_single_image` for details.

        interpolation : None or str or int, optional
            The interpolation to use during resize.
            See :func:`~imgaug.imgaug.imresize_single_image` for details.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Resized heatmaps object.

        """
        arr_0to1_resized = ia.imresize_single_image(
            self.arr_0to1, sizes, interpolation=interpolation)

        # cubic interpolation can lead to values outside of [0.0, 1.0],
        # see https://github.com/opencv/opencv/issues/7195
        # TODO area interpolation too?
        arr_0to1_resized = np.clip(arr_0to1_resized, 0.0, 1.0)

        return HeatmapsOnImage.from_0to1(
            arr_0to1_resized,
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)

    def to_uint8(self):
        """Convert this heatmaps object to an ``uint8`` array.

        Returns
        -------
        (H,W,C) ndarray
            Heatmap as an ``uint8`` array, i.e. with the discrete value
            range ``[0, 255]``.

        """
        # TODO this always returns (H,W,C), even if input ndarray was
        #      originally (H,W). Does it make sense here to also return
        #      (H,W) if self.arr_was_2d?
        arr_0to255 = np.clip(np.round(self.arr_0to1 * 255), 0, 255)
        arr_uint8 = arr_0to255.astype(np.uint8)
        return arr_uint8

    @staticmethod
    def from_uint8(arr_uint8, shape, min_value=0.0, max_value=1.0):
        """Create a ``float``-based heatmaps object from an ``uint8`` array.

        Parameters
        ----------
        arr_uint8 : (H,W) ndarray or (H,W,C) ndarray
            Heatmap(s) array, where ``H`` is height, ``W`` is width
            and ``C`` is the number of heatmap channels.
            Expected dtype is ``uint8``.

        shape : tuple of int
            Shape of the image on which the heatmap(s) is/are placed.
            **Not** the shape of the heatmap(s) array, unless it is identical
            to the image shape (note the likely difference between the arrays
            in the number of channels).
            If there is not a corresponding image, use the shape of the
            heatmaps array.

        min_value : float, optional
            Minimum value of the float heatmaps that the input array
            represents. This will usually be 0.0. In most other cases it will
            be close to the interval ``[0.0, 1.0]``.
            Calling :func:`~imgaug.HeatmapsOnImage.get_arr`, will automatically
            convert the interval ``[0.0, 1.0]`` float array to this
            ``[min, max]`` interval.

        max_value : float, optional
            Minimum value of the float heatmaps that the input array
            represents. This will usually be 1.0.
            See parameter `min_value` for details.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps object.

        """
        arr_0to1 = arr_uint8.astype(np.float32) / 255.0
        return HeatmapsOnImage.from_0to1(
            arr_0to1, shape,
            min_value=min_value,
            max_value=max_value)

    @staticmethod
    def from_0to1(arr_0to1, shape, min_value=0.0, max_value=1.0):
        """Create a heatmaps object from a ``[0.0, 1.0]`` float array.

        Parameters
        ----------
        arr_0to1 : (H,W) or (H,W,C) ndarray
            Heatmap(s) array, where ``H`` is the height, ``W`` is the width
            and ``C`` is the number of heatmap channels.
            Expected dtype is ``float32``.

        shape : tuple of ints
            Shape of the image on which the heatmap(s) is/are placed.
            **Not** the shape of the heatmap(s) array, unless it is identical
            to the image shape (note the likely difference between the arrays
            in the number of channels).
            If there is not a corresponding image, use the shape of the
            heatmaps array.

        min_value : float, optional
            Minimum value of the float heatmaps that the input array
            represents. This will usually be 0.0. In most other cases it will
            be close to the interval ``[0.0, 1.0]``.
            Calling :func:`~imgaug.HeatmapsOnImage.get_arr`, will automatically
            convert the interval ``[0.0, 1.0]`` float array to this
            ``[min, max]`` interval.

        max_value : float, optional
            Minimum value of the float heatmaps that the input array
            represents. This will usually be 1.0.
            See parameter `min_value` for details.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Heatmaps object.

        """
        heatmaps = HeatmapsOnImage(arr_0to1, shape,
                                   min_value=0.0, max_value=1.0)
        heatmaps.min_value = min_value
        heatmaps.max_value = max_value
        return heatmaps

    # TODO change name to change_value_range()?
    @classmethod
    def change_normalization(cls, arr, source, target):
        """Change the value range of a heatmap array.

        E.g. the value range may be changed from the interval ``[0.0, 1.0]``
        to ``[-1.0, 1.0]``.

        Parameters
        ----------
        arr : ndarray
            Heatmap array to modify.

        source : tuple of float
            Current value range of the input array, given as a
            tuple ``(min, max)``, where both are ``float`` values.

        target : tuple of float
            Desired output value range of the array, given as a
            tuple ``(min, max)``, where both are ``float`` values.

        Returns
        -------
        ndarray
            Input array, with value range projected to the desired target
            value range.

        """
        assert ia.is_np_array(arr), (
            "Expected 'arr' to be an ndarray, got type %s." % (type(arr),))

        def _validate_tuple(arg_name, arg_value):
            assert isinstance(arg_value, tuple), (
                "'%s' was not a HeatmapsOnImage instance, "
                "expected type tuple then. Got type %s." % (
                    arg_name, type(arg_value),))
            assert len(arg_value) == 2, (
                "Expected tuple '%s' to contain exactly two entries, "
                "got %d." % (arg_name, len(arg_value),))
            assert arg_value[0] < arg_value[1], (
                "Expected tuple '%s' to have two entries with "
                "entry 1 < entry 2, got values %.4f and %.4f." % (
                    arg_name, arg_value[0], arg_value[1]))

        if isinstance(source, HeatmapsOnImage):
            source = (source.min_value, source.max_value)
        else:
            _validate_tuple("source", source)

        if isinstance(target, HeatmapsOnImage):
            target = (target.min_value, target.max_value)
        else:
            _validate_tuple("target", target)

        # Check if source and target are the same (with a tiny bit of
        # tolerance) if so, evade compuation and just copy the array instead.
        # This is reasonable, as source and target will often both
        # be (0.0, 1.0).
        eps = np.finfo(arr.dtype).eps
        mins_same = source[0] - 10*eps < target[0] < source[0] + 10*eps
        maxs_same = source[1] - 10*eps < target[1] < source[1] + 10*eps
        if mins_same and maxs_same:
            return np.copy(arr)

        min_source, max_source = source
        min_target, max_target = target

        diff_source = max_source - min_source
        diff_target = max_target - min_target

        arr_0to1 = (arr - min_source) / diff_source
        arr_target = min_target + arr_0to1 * diff_target

        return arr_target

    # TODO make this a proper shallow-copy
    def copy(self):
        """Create a shallow copy of the heatmaps object.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Shallow copy.

        """
        return self.deepcopy()

    def deepcopy(self):
        """Create a deep copy of the heatmaps object.

        Returns
        -------
        imgaug.augmentables.heatmaps.HeatmapsOnImage
            Deep copy.

        """
        return HeatmapsOnImage(
            self.get_arr(),
            shape=self.shape,
            min_value=self.min_value,
            max_value=self.max_value)
