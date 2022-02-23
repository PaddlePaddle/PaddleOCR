# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read FITS files.

Backend Library: `Astropy <https://www.astropy.org/>`_

.. note::
    To use this plugin you have to install its backend::

        pip install imageio[fits]

Flexible Image Transport System (FITS) is an open standard defining a
digital file format useful for storage, transmission and processing of
scientific and other images. FITS is the most commonly used digital
file format in astronomy.


Parameters
----------
cache : bool
    If the file name is a URL, `~astropy.utils.data.download_file` is used
    to open the file.  This specifies whether or not to save the file
    locally in Astropy's download cache (default: `True`).
uint : bool
    Interpret signed integer data where ``BZERO`` is the
    central value and ``BSCALE == 1`` as unsigned integer
    data.  For example, ``int16`` data with ``BZERO = 32768``
    and ``BSCALE = 1`` would be treated as ``uint16`` data.

    Note, for backward compatibility, the kwarg **uint16** may
    be used instead.  The kwarg was renamed when support was
    added for integers of any size.
ignore_missing_end : bool
    Do not issue an exception when opening a file that is
    missing an ``END`` card in the last header.
checksum : bool or str
    If `True`, verifies that both ``DATASUM`` and
    ``CHECKSUM`` card values (when present in the HDU header)
    match the header and data of all HDU's in the file.  Updates to a
    file that already has a checksum will preserve and update the
    existing checksums unless this argument is given a value of
    'remove', in which case the CHECKSUM and DATASUM values are not
    checked, and are removed when saving changes to the file.
disable_image_compression : bool, optional
    If `True`, treats compressed image HDU's like normal
    binary table HDU's.
do_not_scale_image_data : bool
    If `True`, image data is not scaled using BSCALE/BZERO values
    when read.
ignore_blank : bool
    If `True`, the BLANK keyword is ignored if present.
scale_back : bool
    If `True`, when saving changes to a file that contained scaled
    image data, restore the data to the original type and reapply the
    original BSCALE/BZERO values.  This could lead to loss of accuracy
    if scaling back to integer values after performing floating point
    operations on the data.

"""

from ..core import Format

_fits = None  # lazily loaded


def load_lib():
    global _fits
    try:
        from astropy.io import fits as _fits
    except ImportError:
        raise ImportError(
            "The FITS format relies on the astropy package."
            "Please refer to http://www.astropy.org/ "
            "for further instructions."
        )
    return _fits


class FitsFormat(Format):
    """See :mod:`imageio.plugins.fits`"""

    def _can_read(self, request):
        # We return True if ext matches, because this is the only plugin
        # that can. If astropy is not installed, a useful error follows.
        return request.extension in self.extensions

    def _can_write(self, request):
        # No write support
        return False

    # -- reader

    class Reader(Format.Reader):
        def _open(self, cache=False, **kwargs):
            if not _fits:
                load_lib()
            hdulist = _fits.open(self.request.get_file(), cache=cache, **kwargs)

            self._index = []
            allowed_hdu_types = (_fits.ImageHDU, _fits.PrimaryHDU, _fits.CompImageHDU)
            for n, hdu in zip(range(len(hdulist)), hdulist):
                if isinstance(hdu, allowed_hdu_types):
                    # Ignore (primary) header units with no data (use '.size'
                    # rather than '.data' to avoid actually loading the image):
                    if hdu.size > 0:
                        self._index.append(n)
            self._hdulist = hdulist

        def _close(self):
            self._hdulist.close()

        def _get_length(self):
            return len(self._index)

        def _get_data(self, index):
            # Get data
            if index < 0 or index >= len(self._index):
                raise IndexError("Index out of range while reading from fits")
            im = self._hdulist[self._index[index]].data
            # Return array and empty meta data
            return im, {}

        def _get_meta_data(self, index):
            # Get the meta data for the given index
            raise RuntimeError("The fits format does not support meta data.")
