# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read GDAL files.

Backend: `GDAL <https://gdal.org/>`_

.. note::
    To use this plugin you have to install its backend::

        pip install imageio[gdal]

Parameters
----------
none
"""

from ..core import Format, has_module

_gdal = None  # lazily loaded in load_lib()


def load_lib():
    global _gdal
    try:
        import osgeo.gdal as _gdal
    except ImportError:
        raise ImportError(
            "The GDAL format relies on the GDAL package."
            "Please refer to http://www.gdal.org/"
            "for further instructions."
        )
    return _gdal


GDAL_FORMATS = (".tiff", " .tif", ".img", ".ecw", ".jpg", ".jpeg")


class GdalFormat(Format):
    """See :mod:`imageio.plugins.gdal`"""

    def _can_read(self, request):
        if request.extension in (".ecw",):
            return True
        if has_module("osgeo.gdal"):
            return request.extension in self.extensions

    def _can_write(self, request):
        return False

    # --

    class Reader(Format.Reader):
        def _open(self):
            if not _gdal:
                load_lib()
            self._ds = _gdal.Open(self.request.get_local_filename())

        def _close(self):
            del self._ds

        def _get_length(self):
            return 1

        def _get_data(self, index):
            if index != 0:
                raise IndexError("Gdal file contains only one dataset")
            return self._ds.ReadAsArray(), self._get_meta_data(index)

        def _get_meta_data(self, index):
            return self._ds.GetMetadata()
