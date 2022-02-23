# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read/Write BSDF files.

Backend Library: internal

The BSDF format enables reading and writing of image data in the
BSDF serialization format. This format allows storage of images, volumes,
and series thereof. Data can be of any numeric data type, and can
optionally be compressed. Each image/volume can have associated
meta data, which can consist of any data type supported by BSDF.

By default, image data is lazily loaded; the actual image data is
not read until it is requested. This allows storing multiple images
in a single file and still have fast access to individual images.
Alternatively, a series of images can be read in streaming mode, reading
images as they are read (e.g. from http).

BSDF is a simple generic binary format. It is easy to extend and there
are standard extension definitions for 2D and 3D image data.
Read more at http://bsdf.io.


Parameters
----------
random_access : bool
    Whether individual images in the file can be read in random order.
    Defaults to True for normal files, and to False when reading from HTTP.
    If False, the file is read in "streaming mode", allowing reading
    files as they are read, but without support for "rewinding".
    Note that setting this to True when reading from HTTP, the whole file
    is read upon opening it (since lazy loading is not possible over HTTP).

compression : int
    Use ``0`` or "no" for no compression, ``1`` or "zlib" for Zlib
    compression (same as zip files and PNG), and ``2`` or "bz2" for Bz2
    compression (more compact but slower). Default 1 (zlib).
    Note that some BSDF implementations may not support compression
    (e.g. JavaScript).

"""

import numpy as np

from ..core import Format


def get_bsdf_serializer(options):
    from . import _bsdf as bsdf

    class NDArrayExtension(bsdf.Extension):
        """Copy of BSDF's NDArrayExtension but deal with lazy blobs."""

        name = "ndarray"
        cls = np.ndarray

        def encode(self, s, v):
            return dict(shape=v.shape, dtype=str(v.dtype), data=v.tobytes())

        def decode(self, s, v):
            return v  # return as dict, because of lazy blobs, decode in Image

    class ImageExtension(bsdf.Extension):
        """We implement two extensions that trigger on the Image classes."""

        def encode(self, s, v):
            return dict(array=v.array, meta=v.meta)

        def decode(self, s, v):
            return Image(v["array"], v["meta"])

    class Image2DExtension(ImageExtension):

        name = "image2d"
        cls = Image2D

    class Image3DExtension(ImageExtension):

        name = "image3d"
        cls = Image3D

    exts = [NDArrayExtension, Image2DExtension, Image3DExtension]
    serializer = bsdf.BsdfSerializer(exts, **options)

    return bsdf, serializer


class Image:
    """Class in which we wrap the array and meta data. By using an extension
    we can make BSDF trigger on these classes and thus encode the images.
    as actual images.
    """

    def __init__(self, array, meta):
        self.array = array
        self.meta = meta

    def get_array(self):
        if not isinstance(self.array, np.ndarray):
            v = self.array
            blob = v["data"]
            if not isinstance(blob, bytes):  # then it's a lazy bsdf.Blob
                blob = blob.get_bytes()
            self.array = np.frombuffer(blob, dtype=v["dtype"])
            self.array.shape = v["shape"]
        return self.array

    def get_meta(self):
        return self.meta


class Image2D(Image):
    pass


class Image3D(Image):
    pass


class BsdfFormat(Format):
    """The BSDF format enables reading and writing of image data in the
    BSDF serialization format. This format allows storage of images, volumes,
    and series thereof. Data can be of any numeric data type, and can
    optionally be compressed. Each image/volume can have associated
    meta data, which can consist of any data type supported by BSDF.

    By default, image data is lazily loaded; the actual image data is
    not read until it is requested. This allows storing multiple images
    in a single file and still have fast access to individual images.
    Alternatively, a series of images can be read in streaming mode, reading
    images as they are read (e.g. from http).

    BSDF is a simple generic binary format. It is easy to extend and there
    are standard extension definitions for 2D and 3D image data.
    Read more at http://bsdf.io.

    Parameters for reading
    ----------------------
    random_access : bool
        Whether individual images in the file can be read in random order.
        Defaults to True for normal files, and to False when reading from HTTP.
        If False, the file is read in "streaming mode", allowing reading
        files as they are read, but without support for "rewinding".
        Note that setting this to True when reading from HTTP, the whole file
        is read upon opening it (since lazy loading is not possible over HTTP).

    Parameters for saving
    ---------------------
    compression : {0, 1, 2}
        Use ``0`` or "no" for no compression, ``1`` or "zlib" for Zlib
        compression (same as zip files and PNG), and ``2`` or "bz2" for Bz2
        compression (more compact but slower). Default 1 (zlib).
        Note that some BSDF implementations may not support compression
        (e.g. JavaScript).

    """

    def _can_read(self, request):
        if request.mode[1] in (self.modes + "?"):
            # if request.extension in self.extensions:
            #     return True
            if request.firstbytes.startswith(b"BSDF"):
                return True

    def _can_write(self, request):
        if request.mode[1] in (self.modes + "?"):
            if request.extension in self.extensions:
                return True

    # -- reader

    class Reader(Format.Reader):
        def _open(self, random_access=None):
            # Validate - we need a BSDF file consisting of a list of images
            # The list is typically a stream, but does not have to be.
            assert self.request.firstbytes[:4] == b"BSDF", "Not a BSDF file"
            # self.request.firstbytes[5:6] == major and minor version
            if not (
                self.request.firstbytes[6:15] == b"M\x07image2D"
                or self.request.firstbytes[6:15] == b"M\x07image3D"
                or self.request.firstbytes[6:7] == b"l"
            ):
                pass  # Actually, follow a more duck-type approach ...
                # raise RuntimeError('BSDF file does not look like an '
                #                   'image container.')
            # Set options. If we think that seeking is allowed, we lazily load
            # blobs, and set streaming to False (i.e. the whole file is read,
            # but we skip over binary blobs), so that we subsequently allow
            # random access to the images.
            # If seeking is not allowed (e.g. with a http request), we cannot
            # lazily load blobs, but we can still load streaming from the web.
            options = {}
            if self.request.filename.startswith(("http://", "https://")):
                ra = False if random_access is None else bool(random_access)
                options["lazy_blob"] = False  # Because we cannot seek now
                options["load_streaming"] = not ra  # Load as a stream?
            else:
                ra = True if random_access is None else bool(random_access)
                options["lazy_blob"] = ra  # Don't read data until needed
                options["load_streaming"] = not ra

            file = self.request.get_file()
            bsdf, self._serializer = get_bsdf_serializer(options)
            self._stream = self._serializer.load(file)
            # Another validation
            if (
                isinstance(self._stream, dict)
                and "meta" in self._stream
                and "array" in self._stream
            ):
                self._stream = Image(self._stream["array"], self._stream["meta"])
            if not isinstance(self._stream, (Image, list, bsdf.ListStream)):
                raise RuntimeError(
                    "BSDF file does not look seem to have an " "image container."
                )

        def _close(self):
            pass

        def _get_length(self):
            if isinstance(self._stream, Image):
                return 1
            elif isinstance(self._stream, list):
                return len(self._stream)
            elif self._stream.count < 0:
                return np.inf
            return self._stream.count

        def _get_data(self, index):
            # Validate
            if index < 0 or index >= self.get_length():
                raise IndexError(
                    "Image index %i not in [0 %i]." % (index, self.get_length())
                )
            # Get Image object
            if isinstance(self._stream, Image):
                image_ob = self._stream  # singleton
            elif isinstance(self._stream, list):
                # Easy when we have random access
                image_ob = self._stream[index]
            else:
                # For streaming, we need to skip over frames
                if index < self._stream.index:
                    raise IndexError(
                        "BSDF file is being read in streaming "
                        "mode, thus does not allow rewinding."
                    )
                while index > self._stream.index:
                    self._stream.next()
                image_ob = self._stream.next()  # Can raise StopIteration
            # Is this an image?
            if (
                isinstance(image_ob, dict)
                and "meta" in image_ob
                and "array" in image_ob
            ):
                image_ob = Image(image_ob["array"], image_ob["meta"])
            if isinstance(image_ob, Image):
                # Return as array (if we have lazy blobs, they are read now)
                return image_ob.get_array(), image_ob.get_meta()
            else:
                r = repr(image_ob)
                r = r if len(r) < 200 else r[:197] + "..."
                raise RuntimeError("BSDF file contains non-image " + r)

        def _get_meta_data(self, index):  # pragma: no cover
            return {}  # This format does not support global meta data

    # -- writer

    class Writer(Format.Writer):
        def _open(self, compression=1):
            options = {"compression": compression}
            bsdf, self._serializer = get_bsdf_serializer(options)
            if self.request.mode[1] in "iv":
                self._stream = None  # Singleton image
                self._written = False
            else:
                # Series (stream) of images
                file = self.request.get_file()
                self._stream = bsdf.ListStream()
                self._serializer.save(file, self._stream)

        def _close(self):
            # We close the stream here, which will mark the number of written
            # elements. If we would not close it, the file would be fine, it's
            # just that upon reading it would not be known how many items are
            # in there.
            if self._stream is not None:
                self._stream.close(False)  # False says "keep this a stream"

        def _append_data(self, im, meta):
            # Determine dimension
            ndim = None
            if self.request.mode[1] in "iI":
                ndim = 2
            elif self.request.mode[1] in "vV":
                ndim = 3
            else:
                ndim = 3  # Make an educated guess
                if im.ndim == 2 or (im.ndim == 3 and im.shape[-1] <= 4):
                    ndim = 2
            # Validate shape
            assert ndim in (2, 3)
            if ndim == 2:
                assert im.ndim == 2 or (im.ndim == 3 and im.shape[-1] <= 4)
            else:
                assert im.ndim == 3 or (im.ndim == 4 and im.shape[-1] <= 4)
            # Wrap data and meta data in our special class that will trigger
            # the BSDF image2D or image3D extension.
            if ndim == 2:
                ob = Image2D(im, meta)
            else:
                ob = Image3D(im, meta)
            # Write directly or to stream
            if self._stream is None:
                assert not self._written, "Cannot write singleton image twice"
                self._written = True
                file = self.request.get_file()
                self._serializer.save(file, ob)
            else:
                self._stream.append(ob)

        def set_meta_data(self, meta):  # pragma: no cover
            raise RuntimeError("The BSDF format only supports " "per-image meta data.")
