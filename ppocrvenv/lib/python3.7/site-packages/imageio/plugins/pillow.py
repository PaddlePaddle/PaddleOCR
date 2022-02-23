# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read/Write images using Pillow/PIL.

Backend Library: `Pillow <https://pillow.readthedocs.io/en/stable/>`_

Plugin that wraps the the Pillow library. Pillow is a friendly fork of PIL
(Python Image Library) and supports reading and writing of common formats (jpg,
png, gif, tiff, ...). For, the complete list of features and supported formats
please refer to pillows official docs (see the Backend Library link).

Parameters
----------
request : Request
    A request object representing the resource to be operated on.

Methods
-------

.. autosummary::
    :toctree: _plugins/pillow

    PillowPlugin.read
    PillowPlugin.write
    PillowPlugin.iter
    PillowPlugin.get_meta

"""

import numpy as np
from PIL import Image, UnidentifiedImageError, ImageSequence, ExifTags
from ..core.request import Request, IOMode, InitializationError, URI_FILE, URI_BYTES


def _is_multichannel(mode: str) -> bool:
    """Returns true if the color mode uses more than one channel.

    We need an easy way to test for this to, for example, figure out which
    dimension to rotate when we encounter an exif rotation flag. I didn't find a
    good way to do this using pillow, so instead we have a local list of all
    (currently) supported modes.

    If somebody comes across a better way to do this, in particular if it
    automatically grabs available modes from pillow a PR is very welcome :)

    Parameters
    ----------
    mode : str
        A valid pillow mode string

    Returns
    -------
    is_multichannel : bool
        True if the image uses more than one channel to represent a color, False
        otherwise.

    """
    multichannel = {
        "BGR;15": True,
        "BGR;16": True,
        "BGR;24": True,
        "BGR;32": True,
    }

    if mode in multichannel:
        return multichannel[mode]

    return Image.getmodebands(mode) > 1


def _exif_orientation_transform(orientation, mode):
    # get transformation that transforms an image from a
    # given EXIF orientation into the standard orientation

    # -1 if the mode has color channel, 0 otherwise
    axis = -2 if _is_multichannel(mode) else -1

    EXIF_ORIENTATION = {
        1: lambda x: x,
        2: lambda x: np.flip(x, axis=axis),
        3: lambda x: np.rot90(x, k=2),
        4: lambda x: np.flip(x, axis=axis - 1),
        5: lambda x: np.flip(np.rot90(x, k=3), axis=axis),
        6: lambda x: np.rot90(x, k=1),
        7: lambda x: np.flip(np.rot90(x, k=1), axis=axis),
        8: lambda x: np.rot90(x, k=3),
    }

    return EXIF_ORIENTATION[orientation]


class PillowPlugin(object):
    def __init__(self, request: Request):
        """Instantiate a new Pillow Plugin Object

        Parameters
        ----------
        request : {Request}
            A request object representing the resource to be operated on.

        """

        self._request = request
        self._image = None

        if request.mode.io_mode == IOMode.read:
            try:
                with Image.open(request.get_file()):
                    # Check if it is generally possible to read the image.
                    # This will not read any data and merely try to find a
                    # compatible pillow plugin (ref: the pillow docs).
                    pass
            except UnidentifiedImageError:
                if request._uri_type == URI_BYTES:
                    raise InitializationError(
                        "Pillow can not read the provided bytes."
                    ) from None
                else:
                    raise InitializationError(
                        f"Pillow can not read {request.raw_uri}."
                    ) from None
        else:
            # it would be nice if pillow would expose a
            # function to check if an extension can be written
            # instead of us digging in the internals

            Image.preinit()
            if request.extension not in Image.EXTENSION:
                Image.init()

            if request._uri_type == URI_FILE:
                pass  # OK, is file-like object
            elif request._uri_type == URI_BYTES:
                pass  # OK, is bytes string
            elif request.extension in Image.EXTENSION:
                pass  # OK, is extension known to pillow
            else:
                raise InitializationError(
                    f"Pillow can not write `{request.extension}` files"
                ) from None

        if self._request.mode.io_mode == IOMode.read:
            self._image = Image.open(self._request.get_file())

    def close(self):
        if self._image:
            self._image.close()

        self._request.finish()

    def read(self, *, index=None, mode=None, rotate=False, apply_gamma=False):
        """
        Parses the given URI and creates a ndarray from it.

        Parameters
        ----------
        index : {integer}
            If the URI contains a list of ndimages (multiple frames) return the
            index-th image/frame. If None, read all ndimages (frames) in the URI
            and attempt to stack them along a new 0-th axis (equivalent to
            np.stack(imgs, axis=0))
        mode : {str, None}
            Convert the image to the given mode before returning it. If None,
            the mode will be left unchanged. Possible modes can be found at:
            https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
        rotate : {bool}
            If set to ``True`` and the image contains an EXIF orientation tag,
            apply the orientation before returning the ndimage.
        apply_gamma : {bool}
            If ``True`` and the image contains metadata about gamma, apply gamma
            correction to the image.

        Returns
        -------
        ndimage : ndarray
            A numpy array containing the loaded image data

        Notes
        -----
        If you open a GIF - or any other format using color pallets - you may
        wish to manually set the `mode` parameter. Otherwise, the numbers in
        the returned image will refer to the entries in the color pallet, which
        is discarded during conversion to ndarray.

        """

        if index is not None:
            # will raise IO error if index >= number of frames in image
            self._image.seek(index)
            image = self._apply_transforms(self._image, mode, rotate, apply_gamma)
            return image
        else:
            iterator = self.iter(mode=mode, rotate=rotate, apply_gamma=apply_gamma)
            image = np.stack([im for im in iterator], axis=0)
            if image.shape[0] == 1:
                image = np.squeeze(image, axis=0)
            return image

    def iter(self, *, mode=None, rotate=False, apply_gamma=False):
        """
        Iterate over all ndimages/frames in the URI

        Parameters
        ----------
        mode : {str, None}
            Convert the image to the given mode before returning it. If None,
            the mode will be left unchanged. Possible modes can be found at:
            https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
        rotate : {bool}
            If set to ``True`` and the image contains an EXIF orientation tag,
            apply the orientation before returning the ndimage.
        apply_gamma : {bool}
            If ``True`` and the image contains metadata about gamma, apply gamma
            correction to the image.
        """

        for im in ImageSequence.Iterator(self._image):
            yield self._apply_transforms(im, mode, rotate, apply_gamma)

    def _apply_transforms(self, image, mode, rotate, apply_gamma):
        if mode is not None:
            image = image.convert(mode)
        elif image.format == "GIF":
            # adjust for pillow9 changes
            # see: https://github.com/python-pillow/Pillow/issues/5929
            image = image.convert(image.palette.mode)
        image = np.asarray(image)

        meta = self.get_meta()
        if rotate and "Orientation" in meta:
            transformation = _exif_orientation_transform(
                meta["Orientation"], self._image.mode
            )
            image = transformation(image)

        if apply_gamma and "gamma" in meta:
            gamma = float(meta["gamma"])
            scale = float(65536 if image.dtype == np.uint16 else 255)
            gain = 1.0
            image = ((image / scale) ** gamma) * scale * gain + 0.4999
            image = np.round(image).astype(np.uint8)

        return image

    def write(self, image: np.ndarray, *, mode=None, format=None, **kwargs):
        """
        Write an ndimage to the URI specified in path.

        If the URI points to a file on the current host and the file does not
        yet exist it will be created. If the file exists already, it will be
        appended if possible; otherwise, it will be replaced.

        If necessary, the image is broken down along the leading dimension to
        fit into individual frames of the chosen format. If the format doesn't
        support multiple frames, and IOError is raised.

        Parameters
        ----------
        image : ndarray
            The ndimage to write.
        mode : {str, None}
            Specify the image's color format. If None (default), the mode is
            inferred from the array's shape and dtype. Possible modes can be
            found at:
            https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes
        format : {str, None}
            Optional format override.  If omitted, the format to use is
            determined from the filename extension. If a file object was used
            instead of a filename, this parameter must always be used.
        kwargs : ...
            Extra arguments to pass to pillow. If a writer doesn't recognise an
            option, it is silently ignored. The available options are described
            in pillow's `image format documentation
            <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_
            for each writer.

        Notes
        -----
        When writing batches of very narrow (2-4 pixels wide) gray images set
        the ``mode`` explicitly to avoid the batch being identified as a colored
        image.

        """

        save_args = {
            "format": format,
        }

        # check if ndimage is a batch of frames/pages (e.g. for writing GIF)
        # if mode is given, use it; otherwise fall back to image.ndim only
        if mode is not None:
            is_batch = image.ndim > 3 if _is_multichannel(mode) else image.ndim > 2
        elif image.ndim == 2:
            is_batch = False
        elif image.ndim == 3 and image.shape[-1] in [2, 3, 4]:
            # Note: this makes a channel-last assumption
            # (pillow seems to make it as well)
            is_batch = False
        else:
            is_batch = True

        if is_batch:
            save_args["save_all"] = True
            primary_image = Image.fromarray(image[0], mode=mode)

            append_images = list()
            for frame in image[1:]:
                pil_frame = Image.fromarray(frame, mode=mode)
                if "bits" in kwargs:
                    pil_frame = pil_frame.quantize(colors=2 ** kwargs["bits"])
                append_images.append(pil_frame)
            save_args["append_images"] = append_images
        else:
            primary_image = Image.fromarray(image, mode=mode)

        if "bits" in kwargs:
            primary_image = primary_image.quantize(colors=2 ** kwargs["bits"])

        save_args.update(kwargs)
        primary_image.save(self._request.get_file(), **save_args)

        if self._request._uri_type == URI_BYTES:
            return self._request.get_file().getvalue()

    def get_meta(self, *, index=None):
        """Read ndimage metadata from the URI

        Parameters
        ----------
        index : {integer, None}
            If the URI contains a list of ndimages return the metadata
            corresponding to the index-th image. If None, return the metadata
            for the last read ndimage/frame.
        """

        if index is not None:
            self._image.seek(index)

        metadata = self._image.info

        if self._image.mode == "P":
            metadata["palette"] = self._image.palette

        if self._image.getexif():
            exif_data = {
                ExifTags.TAGS.get(key, "unknown"): value
                for key, value in dict(self._image.getexif()).items()
            }
            exif_data.pop("unknown", None)
            metadata.update(exif_data)

        return self._image.info

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self) -> None:
        self.close()
