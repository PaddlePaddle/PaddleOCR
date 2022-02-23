# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Plugin for multi-image freeimafe formats, like animated GIF and ico.
"""

import logging
import numpy as np

from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat

logger = logging.getLogger(__name__)


class FreeimageMulti(FreeimageFormat):
    """Base class for freeimage formats that support multiple images."""

    _modes = "iI"
    _fif = -1

    class Reader(Format.Reader):
        def _open(self, flags=0):
            flags = int(flags)
            # Create bitmap
            self._bm = fi.create_multipage_bitmap(
                self.request.filename, self.format.fif, flags
            )
            self._bm.load_from_filename(self.request.get_local_filename())

        def _close(self):
            self._bm.close()

        def _get_length(self):
            return len(self._bm)

        def _get_data(self, index):
            sub = self._bm.get_page(index)
            try:
                return sub.get_image_data(), sub.get_meta_data()
            finally:
                sub.close()

        def _get_meta_data(self, index):
            index = index or 0
            if index < 0 or index >= len(self._bm):
                raise IndexError()
            sub = self._bm.get_page(index)
            try:
                return sub.get_meta_data()
            finally:
                sub.close()

    # --

    class Writer(FreeimageFormat.Writer):
        def _open(self, flags=0):
            # Set flags
            self._flags = flags = int(flags)
            # Instantiate multi-page bitmap
            self._bm = fi.create_multipage_bitmap(
                self.request.filename, self.format.fif, flags
            )
            self._bm.save_to_filename(self.request.get_local_filename())

        def _close(self):
            # Close bitmap
            self._bm.close()

        def _append_data(self, im, meta):
            # Prepare data
            if im.ndim == 3 and im.shape[-1] == 1:
                im = im[:, :, 0]
            im = image_as_uint(im, bitdepth=8)
            # Create sub bitmap
            sub1 = fi.create_bitmap(self._bm._filename, self.format.fif)
            # Let subclass add data to bitmap, optionally return new
            sub2 = self._append_bitmap(im, meta, sub1)
            # Add
            self._bm.append_bitmap(sub2)
            sub2.close()
            if sub1 is not sub2:
                sub1.close()

        def _append_bitmap(self, im, meta, bitmap):
            # Set data
            bitmap.allocate(im)
            bitmap.set_image_data(im)
            bitmap.set_meta_data(meta)
            # Return that same bitmap
            return bitmap

        def _set_meta_data(self, meta):
            pass  # ignore global meta data


class MngFormat(FreeimageMulti):
    """An Mng format based on the Freeimage library.

    Read only. Seems broken.
    """

    _fif = 6

    def _can_write(self, request):  # pragma: no cover
        return False


class IcoFormat(FreeimageMulti):
    """An ICO format based on the Freeimage library.

    This format supports grayscale, RGB and RGBA images.

    The freeimage plugin requires a `freeimage` binary. If this binary
    is not available on the system, it can be downloaded by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for reading
    ----------------------
    makealpha : bool
        Convert to 32-bit and create an alpha channel from the AND-
        mask when loading. Default False. Note that this returns wrong
        results if the image was already RGBA.

    """

    _fif = 1

    class Reader(FreeimageMulti.Reader):
        def _open(self, flags=0, makealpha=False):
            # Build flags from kwargs
            flags = int(flags)
            if makealpha:
                flags |= IO_FLAGS.ICO_MAKEALPHA
            return FreeimageMulti.Reader._open(self, flags)


class GifFormat(FreeimageMulti):
    """A format for reading and writing static and animated GIF, based
    on the Freeimage library.

    Images read with this format are always RGBA. Currently,
    the alpha channel is ignored when saving RGB images with this
    format.

    The freeimage plugin requires a `freeimage` binary. If this binary
    is not available on the system, it can be downloaded by either

    - the command line script ``imageio_download_bin freeimage``
    - the Python method ``imageio.plugins.freeimage.download()``

    Parameters for reading
    ----------------------
    playback : bool
        'Play' the GIF to generate each frame (as 32bpp) instead of
        returning raw frame data when loading. Default True.

    Parameters for saving
    ---------------------
    loop : int
        The number of iterations. Default 0 (meaning loop indefinitely)
    duration : {float, list}
        The duration (in seconds) of each frame. Either specify one value
        that is used for all frames, or one value for each frame.
        Note that in the GIF format the duration/delay is expressed in
        hundredths of a second, which limits the precision of the duration.
    fps : float
        The number of frames per second. If duration is not given, the
        duration for each frame is set to 1/fps. Default 10.
    palettesize : int
        The number of colors to quantize the image to. Is rounded to
        the nearest power of two. Default 256.
    quantizer : {'wu', 'nq'}
        The quantization algorithm:
            * wu - Wu, Xiaolin, Efficient Statistical Computations for
              Optimal Color Quantization
            * nq (neuqant) - Dekker A. H., Kohonen neural networks for
              optimal color quantization
    subrectangles : bool
        If True, will try and optimize the GIF by storing only the
        rectangular parts of each frame that change with respect to the
        previous. Unfortunately, this option seems currently broken
        because FreeImage does not handle DisposalMethod correctly.
        Default False.
    """

    _fif = 25

    class Reader(FreeimageMulti.Reader):
        def _open(self, flags=0, playback=True):
            # Build flags from kwargs
            flags = int(flags)
            if playback:
                flags |= IO_FLAGS.GIF_PLAYBACK
            FreeimageMulti.Reader._open(self, flags)

        def _get_data(self, index):
            im, meta = FreeimageMulti.Reader._get_data(self, index)
            # im = im[:, :, :3]  # Drop alpha channel
            return im, meta

    # -- writer

    class Writer(FreeimageMulti.Writer):

        # todo: subrectangles
        # todo: global palette

        def _open(
            self,
            flags=0,
            loop=0,
            duration=None,
            fps=10,
            palettesize=256,
            quantizer="Wu",
            subrectangles=False,
        ):
            # Check palettesize
            if palettesize < 2 or palettesize > 256:
                raise ValueError("GIF quantize param must be 2..256")
            if palettesize not in [2, 4, 8, 16, 32, 64, 128, 256]:
                palettesize = 2 ** int(np.log2(128) + 0.999)
                logger.warning(
                    "Warning: palettesize (%r) modified to a factor of "
                    "two between 2-256." % palettesize
                )
            self._palettesize = palettesize
            # Check quantizer
            self._quantizer = {"wu": 0, "nq": 1}.get(quantizer.lower(), None)
            if self._quantizer is None:
                raise ValueError('Invalid quantizer, must be "wu" or "nq".')
            # Check frametime
            if duration is None:
                self._frametime = [int(1000 / float(fps) + 0.5)]
            elif isinstance(duration, list):
                self._frametime = [int(1000 * d) for d in duration]
            elif isinstance(duration, (float, int)):
                self._frametime = [int(1000 * duration)]
            else:
                raise ValueError("Invalid value for duration: %r" % duration)
            # Check subrectangles
            self._subrectangles = bool(subrectangles)
            self._prev_im = None
            # Init
            FreeimageMulti.Writer._open(self, flags)
            # Set global meta data
            self._meta = {}
            self._meta["ANIMATION"] = {
                # 'GlobalPalette': np.array([0]).astype(np.uint8),
                "Loop": np.array([loop]).astype(np.uint32),
                # 'LogicalWidth': np.array([x]).astype(np.uint16),
                # 'LogicalHeight': np.array([x]).astype(np.uint16),
            }

        def _append_bitmap(self, im, meta, bitmap):
            # Prepare meta data
            meta = meta.copy()
            meta_a = meta["ANIMATION"] = {}
            # If this is the first frame, assign it our "global" meta data
            if len(self._bm) == 0:
                meta.update(self._meta)
                meta_a = meta["ANIMATION"]
            # Set frame time
            index = len(self._bm)
            if index < len(self._frametime):
                ft = self._frametime[index]
            else:
                ft = self._frametime[-1]
            meta_a["FrameTime"] = np.array([ft]).astype(np.uint32)
            # Check array
            if im.ndim == 3 and im.shape[-1] == 4:
                im = im[:, :, :3]
            # Process subrectangles
            im_uncropped = im
            if self._subrectangles and self._prev_im is not None:
                im, xy = self._get_sub_rectangles(self._prev_im, im)
                meta_a["DisposalMethod"] = np.array([1]).astype(np.uint8)
                meta_a["FrameLeft"] = np.array([xy[0]]).astype(np.uint16)
                meta_a["FrameTop"] = np.array([xy[1]]).astype(np.uint16)
            self._prev_im = im_uncropped
            # Set image data
            sub2 = sub1 = bitmap
            sub1.allocate(im)
            sub1.set_image_data(im)
            # Quantize it if its RGB
            if im.ndim == 3 and im.shape[-1] == 3:
                sub2 = sub1.quantize(self._quantizer, self._palettesize)
            # If single image, omit animation data
            if self.request.mode[1] == "i":
                del meta["ANIMATION"]
            # Set meta data and return
            sub2.set_meta_data(meta)
            return sub2

        def _get_sub_rectangles(self, prev, im):
            """
            Calculate the minimal rectangles that need updating each frame.
            Returns a two-element tuple containing the cropped images and a
            list of x-y positions.
            """
            # Get difference, sum over colors
            diff = np.abs(im - prev)
            if diff.ndim == 3:
                diff = diff.sum(2)
            # Get begin and end for both dimensions
            X = np.argwhere(diff.sum(0))
            Y = np.argwhere(diff.sum(1))
            # Get rect coordinates
            if X.size and Y.size:
                x0, x1 = int(X[0]), int(X[-1]) + 1
                y0, y1 = int(Y[0]), int(Y[-1]) + 1
            else:  # No change ... make it minimal
                x0, x1 = 0, 2
                y0, y1 = 0, 2
            # Cut out and return
            return im[y0:y1, x0:x1], (x0, y0)
