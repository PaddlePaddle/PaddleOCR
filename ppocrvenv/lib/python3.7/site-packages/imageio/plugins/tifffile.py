# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read/Write TIFF files.

Backend: internal

Provides support for a wide range of Tiff images using the tifffile
backend.

Parameters for reading
----------------------
offset : int
    Optional start position of embedded file. By default this is
    the current file position.
size : int
    Optional size of embedded file. By default this is the number
    of bytes from the 'offset' to the end of the file.
multifile : bool
    If True (default), series may include pages from multiple files.
    Currently applies to OME-TIFF only.
multifile_close : bool
    If True (default), keep the handles of other files in multifile
    series closed. This is inefficient when few files refer to
    many pages. If False, the C runtime may run out of resources.

Parameters for saving
---------------------
bigtiff : bool
    If True, the BigTIFF format is used.
byteorder : {'<', '>'}
    The endianness of the data in the file.
    By default this is the system's native byte order.
software : str
    Name of the software used to create the image.
    Saved with the first page only.

Metadata for reading
--------------------
planar_configuration : {'contig', 'planar'}
    Specifies if samples are stored contiguous or in separate planes.
    By default this setting is inferred from the data shape.
    'contig': last dimension contains samples.
    'planar': third last dimension contains samples.
resolution_unit : (float, float) or ((int, int), (int, int))
    X and Y resolution in dots per inch as float or rational numbers.
compression : int
    Value indicating the compression algorithm used, e.g. 5 is LZW,
    7 is JPEG, 8 is deflate.
    If 1, data are uncompressed.
predictor : int
    Value 2 indicates horizontal differencing was used before compression,
    while 3 indicates floating point horizontal differencing.
    If 1, no prediction scheme was used before compression.
orientation : {'top_left', 'bottom_right', ...}
    Oriented of image array.
is_rgb : bool
    True if page contains a RGB image.
is_contig : bool
    True if page contains a contiguous image.
is_tiled : bool
    True if page contains tiled image.
is_palette : bool
    True if page contains a palette-colored image and not OME or STK.
is_reduced : bool
    True if page is a reduced image of another image.
is_shaped : bool
    True if page contains shape in image_description tag.
is_fluoview : bool
    True if page contains FluoView MM_STAMP tag.
is_nih : bool
    True if page contains NIH image header.
is_micromanager : bool
    True if page contains Micro-Manager metadata.
is_ome : bool
    True if page contains OME-XML in image_description tag.
is_sgi : bool
    True if page contains SGI image and tile depth tags.
is_stk : bool
    True if page contains UIC2Tag tag.
is_mdgel : bool
    True if page contains md_file_tag tag.
is_mediacy : bool
    True if page contains Media Cybernetics Id tag.
is_stk : bool
    True if page contains UIC2Tag tag.
is_lsm : bool
    True if page contains LSM CZ_LSM_INFO tag.
description : str
    Image description
description1 : str
    Additional description
is_imagej : None or str
    ImageJ metadata
software : str
    Software used to create the TIFF file
datetime : datetime.datetime
    Creation date and time

Metadata for writing
--------------------
photometric : {'minisblack', 'miniswhite', 'rgb'}
    The color space of the image data.
    By default this setting is inferred from the data shape.
planarconfig : {'contig', 'planar'}
    Specifies if samples are stored contiguous or in separate planes.
    By default this setting is inferred from the data shape.
    'contig': last dimension contains samples.
    'planar': third last dimension contains samples.
resolution : (float, float) or ((int, int), (int, int))
    X and Y resolution in dots per inch as float or rational numbers.
description : str
    The subject of the image. Saved with the first page only.
compress : int
    Values from 0 to 9 controlling the level of zlib (deflate) compression.
    If 0, data are written uncompressed (default).
predictor : bool
    If True, horizontal differencing is applied before compression.
    Note that using an int literal 1 actually means no prediction scheme
    will be used.
volume : bool
    If True, volume data are stored in one tile (if applicable) using
    the SGI image_depth and tile_depth tags.
    Image width and depth must be multiple of 16.
    Few software can read this format, e.g. MeVisLab.
writeshape : bool
    If True, write the data shape to the image_description tag
    if necessary and no other description is given.
extratags: sequence of tuples
    Additional tags as [(code, dtype, count, value, writeonce)].

    code : int
        The TIFF tag Id.
    dtype : str
        Data type of items in 'value' in Python struct format.
        One of B, s, H, I, 2I, b, h, i, f, d, Q, or q.
    count : int
        Number of data values. Not used for string values.
    value : sequence
        'Count' values compatible with 'dtype'.
    writeonce : bool
        If True, the tag is written to the first page only.

Notes
-----
Global metadata is stored with the first frame in a TIFF file.
Thus calling :py:meth:`Format.Writer.set_meta_data` after the first frame
was written has no effect. Also, global metadata is ignored if metadata is
provided via the `meta` argument of :py:meth:`Format.Writer.append_data`.

If you have installed tifffile as a Python package, imageio will attempt
to use that as backend instead of the bundled backend. Doing so can
provide access to new performance improvements and bug fixes.

"""

import datetime

from ..core import Format
from ..core.request import URI_BYTES, URI_FILE

import numpy as np


try:
    import tifffile as _tifffile
except ImportError:
    from . import _tifffile


TIFF_FORMATS = (".tif", ".tiff", ".stk", ".lsm")
WRITE_METADATA_KEYS = (
    "photometric",
    "planarconfig",
    "resolution",
    "description",
    "compress",
    "predictor",
    "volume",
    "writeshape",
    "extratags",
    "datetime",
)
READ_METADATA_KEYS = (
    "planar_configuration",
    "is_fluoview",
    "is_nih",
    "is_contig",
    "is_micromanager",
    "is_ome",
    "is_lsm" "is_palette",
    "is_reduced",
    "is_rgb",
    "is_sgi",
    "is_shaped",
    "is_stk",
    "is_tiled",
    "is_mdgel" "resolution_unit",
    "compression",
    "predictor",
    "is_mediacy",
    "orientation",
    "description",
    "description1",
    "is_imagej",
    "software",
)


class TiffFormat(Format):
    """Provides support for a wide range of Tiff images using the tifffile
    backend.

    Images that contain multiple pages can be read using ``imageio.mimread()``
    to read the individual pages, or ``imageio.volread()`` to obtain a
    single (higher dimensional) array.

    Note that global metadata is stored with the first frame in a TIFF file.
    Thus calling :py:meth:`Format.Writer.set_meta_data` after the first frame
    was written has no effect. Also, global metadata is ignored if metadata is
    provided via the `meta` argument of :py:meth:`Format.Writer.append_data`.

    If you have installed tifffile as a Python package, imageio will attempt
    to use that as backend instead of the bundled backend. Doing so can
    provide access to new performance improvements and bug fixes.

    Parameters for reading
    ----------------------
    offset : int
        Optional start position of embedded file. By default this is
        the current file position.
    size : int
        Optional size of embedded file. By default this is the number
        of bytes from the 'offset' to the end of the file.
    multifile : bool
        If True (default), series may include pages from multiple files.
        Currently applies to OME-TIFF only.
    multifile_close : bool
        If True (default), keep the handles of other files in multifile
        series closed. This is inefficient when few files refer to
        many pages. If False, the C runtime may run out of resources.

    Parameters for saving
    ---------------------
    bigtiff : bool
        If True, the BigTIFF format is used.
    byteorder : {'<', '>'}
        The endianness of the data in the file.
        By default this is the system's native byte order.
    software : str
        Name of the software used to create the image.
        Saved with the first page only.

    Metadata for reading
    --------------------
    planar_configuration : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution_unit : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    compression : int
        Value indicating the compression algorithm used, e.g. 5 is LZW,
        7 is JPEG, 8 is deflate.
        If 1, data are uncompressed.
    predictor : int
        Value 2 indicates horizontal differencing was used before compression,
        while 3 indicates floating point horizontal differencing.
        If 1, no prediction scheme was used before compression.
    orientation : {'top_left', 'bottom_right', ...}
        Oriented of image array.
    is_rgb : bool
        True if page contains a RGB image.
    is_contig : bool
        True if page contains a contiguous image.
    is_tiled : bool
        True if page contains tiled image.
    is_palette : bool
        True if page contains a palette-colored image and not OME or STK.
    is_reduced : bool
        True if page is a reduced image of another image.
    is_shaped : bool
        True if page contains shape in image_description tag.
    is_fluoview : bool
        True if page contains FluoView MM_STAMP tag.
    is_nih : bool
        True if page contains NIH image header.
    is_micromanager : bool
        True if page contains Micro-Manager metadata.
    is_ome : bool
        True if page contains OME-XML in image_description tag.
    is_sgi : bool
        True if page contains SGI image and tile depth tags.
    is_stk : bool
        True if page contains UIC2Tag tag.
    is_mdgel : bool
        True if page contains md_file_tag tag.
    is_mediacy : bool
        True if page contains Media Cybernetics Id tag.
    is_stk : bool
        True if page contains UIC2Tag tag.
    is_lsm : bool
        True if page contains LSM CZ_LSM_INFO tag.
    description : str
        Image description
    description1 : str
        Additional description
    is_imagej : None or str
        ImageJ metadata
    software : str
        Software used to create the TIFF file
    datetime : datetime.datetime
        Creation date and time

    Metadata for writing
    --------------------
    photometric : {'minisblack', 'miniswhite', 'rgb'}
        The color space of the image data.
        By default this setting is inferred from the data shape.
    planarconfig : {'contig', 'planar'}
        Specifies if samples are stored contiguous or in separate planes.
        By default this setting is inferred from the data shape.
        'contig': last dimension contains samples.
        'planar': third last dimension contains samples.
    resolution : (float, float) or ((int, int), (int, int))
        X and Y resolution in dots per inch as float or rational numbers.
    description : str
        The subject of the image. Saved with the first page only.
    compress : int
        Values from 0 to 9 controlling the level of zlib (deflate) compression.
        If 0, data are written uncompressed (default).
    predictor : bool
        If True, horizontal differencing is applied before compression.
        Note that using an int literal 1 actually means no prediction scheme
        will be used.
    volume : bool
        If True, volume data are stored in one tile (if applicable) using
        the SGI image_depth and tile_depth tags.
        Image width and depth must be multiple of 16.
        Few software can read this format, e.g. MeVisLab.
    writeshape : bool
        If True, write the data shape to the image_description tag
        if necessary and no other description is given.
    extratags: sequence of tuples
        Additional tags as [(code, dtype, count, value, writeonce)].

        code : int
            The TIFF tag Id.
        dtype : str
            Data type of items in 'value' in Python struct format.
            One of B, s, H, I, 2I, b, h, i, f, d, Q, or q.
        count : int
            Number of data values. Not used for string values.
        value : sequence
            'Count' values compatible with 'dtype'.
        writeonce : bool
            If True, the tag is written to the first page only.
    """

    def _can_read(self, request):
        try:
            _tifffile.TiffFile(request.get_file(), **request.kwargs)
        except ValueError:
            # vendored backend raises value exception
            return False
        except _tifffile.TiffFileError:  # pragma: no-cover
            # current version raises custom exception
            return False
        finally:
            request.get_file().seek(0)

        return True

    def _can_write(self, request):
        if request._uri_type in [URI_FILE, URI_BYTES]:
            pass  # special URI
        elif request.extension not in self.extensions:
            return False

        try:
            _tifffile.TiffWriter(request.get_file(), **request.kwargs)
        except ValueError:
            # vendored backend raises value exception
            return False
        except _tifffile.TiffFileError:  # pragma: no-cover
            # current version raises custom exception
            return False
        finally:
            request.get_file().seek(0)
        return True

    # -- reader

    class Reader(Format.Reader):
        def _open(self, **kwargs):
            # Allow loading from http; tifffile uses seek, so download first
            if self.request.filename.startswith(("http://", "https://")):
                self._f = f = open(self.request.get_local_filename(), "rb")
            else:
                self._f = None
                f = self.request.get_file()
            self._tf = _tifffile.TiffFile(f, **kwargs)

        def _close(self):
            self._tf.close()
            if self._f is not None:
                self._f.close()

        def _get_length(self):
            if self.request.mode[1] in "vV":
                return len(self._tf.series)
            else:
                # for backwards compatibility
                # imread/mimread reads all pages individually
                return len(self._tf.pages)

        def _get_data(self, index):
            if self.request.mode[1] in "vV":
                # this might regress #559 / #558
                # but it is hard to test without a test image
                im = self._tf.asarray(series=index)
            else:
                # Read as 2D image
                if index < 0 or index >= self._get_length():
                    raise IndexError("Index out of range while reading from tiff file")
                im = self._tf.pages[index].asarray()
            # Return array and empty meta data
            meta = self._get_meta_data(index)
            return im, meta

        def _get_meta_data(self, index):
            meta = {}
            page = self._tf.pages[index or 0]
            for key in READ_METADATA_KEYS:
                try:
                    meta[key] = getattr(page, key)
                except Exception:
                    pass

            # tifffile <= 0.12.1 use datetime, newer use DateTime
            for key in ("datetime", "DateTime"):
                try:
                    meta["datetime"] = datetime.datetime.strptime(
                        page.tags[key].value, "%Y:%m:%d %H:%M:%S"
                    )
                    break
                except Exception:
                    pass

            return meta

    # -- writer
    class Writer(Format.Writer):
        def _open(self, bigtiff=None, byteorder=None, software=None):
            try:
                self._tf = _tifffile.TiffWriter(
                    self.request.get_file(), bigtiff, byteorder, software=software
                )
                self._software = None
            except TypeError:
                # In tifffile >= 0.15, the `software` arg is passed to
                # TiffWriter.save
                self._tf = _tifffile.TiffWriter(
                    self.request.get_file(), bigtiff, byteorder
                )
                self._software = software

            self._meta = {}
            self._frames_written = 0

        def _close(self):
            self._tf.close()

        def _append_data(self, im, meta):
            if meta is not None:
                meta = self._sanitize_meta(meta)
            else:
                # Use global metadata for first frame
                meta = self._meta if self._frames_written == 0 else {}
            if self._software is not None and self._frames_written == 0:
                meta["software"] = self._software
            # No need to check self.request.mode; tifffile figures out whether
            # this is a single page, or all page data at once.
            try:
                # TiffWriter.save has been deprecated in version 2020.9.30
                write_meth = self._tf.write
            except AttributeError:
                write_meth = self._tf.save
            write_meth(np.asanyarray(im), contiguous=False, **meta)
            self._frames_written += 1

        @staticmethod
        def _sanitize_meta(meta):
            ret = {}
            for (key, value) in meta.items():
                if key in WRITE_METADATA_KEYS:
                    # Special case of previously read `predictor` int value
                    # 1(=NONE) translation to False expected by TiffWriter.save
                    if key == "predictor" and not isinstance(value, bool):
                        ret[key] = value > 1
                    else:
                        ret[key] = value
            return ret

        def set_meta_data(self, meta):
            self._meta = self._sanitize_meta(meta)
