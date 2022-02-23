# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read/Write images using SimpleITK.

Backend: `Insight Toolkit <https://itk.org/>`_

.. note::
    To use this plugin you have to install its backend::

        pip install imageio[itk]

The ItkFormat uses the ITK or SimpleITK library to support a range of
ITK-related formats. It also supports a few common formats (e.g. PNG and JPEG).

Parameters
----------
None

"""

from ..core import Format, has_module

_itk = None  # Defer loading to load_lib() function.


def load_lib():
    global _itk, _read_function, _write_function
    try:
        import itk as _itk

        _read_function = _itk.imread
        _write_function = _itk.imwrite
    except ImportError:
        try:
            import SimpleITK as _itk

            _read_function = _itk.ReadImage
            _write_function = _itk.WriteImage
        except ImportError:
            raise ImportError(
                "itk could not be found. "
                "Please try "
                "  python -m pip install itk "
                "or "
                "  python -m pip install simpleitk "
                "or refer to "
                "  https://itkpythonpackage.readthedocs.io/ "
                "for further instructions."
            )
    return _itk


# Split up in real ITK and all supported formats.
ITK_FORMATS = (
    ".gipl",
    ".ipl",
    ".mha",
    ".mhd",
    ".nhdr",
    "nia",
    "hdr",
    ".nrrd",
    ".nii",
    ".nii.gz",
    ".img",
    ".img.gz",
    ".vtk",
    "hdf5",
    "lsm",
    "mnc",
    "mnc2",
    "mgh",
    "mnc",
    "pic",
)
ALL_FORMATS = ITK_FORMATS + (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tiff",
    ".tif",
    ".dicom",
    ".dcm",
    ".gdcm",
)


class ItkFormat(Format):
    """See :mod:`imageio.plugins.simpleitk`"""

    def _can_read(self, request):
        # If the request is a format that only this plugin can handle,
        # we report that we can do it; a useful error will be raised
        # when simpleitk is not installed. For the more common formats
        # we only report that we can read if the library is installed.
        if request.extension in ITK_FORMATS:
            return True
        if has_module("itk.ImageIOBase") or has_module("SimpleITK"):
            return request.extension in ALL_FORMATS

    def _can_write(self, request):
        if request.extension in ITK_FORMATS:
            return True
        if has_module("itk.ImageIOBase") or has_module("SimpleITK"):
            return request.extension in ALL_FORMATS

    # -- reader

    class Reader(Format.Reader):
        def _open(self, pixel_type=None, fallback_only=None, **kwargs):
            if not _itk:
                load_lib()
            args = ()
            if pixel_type is not None:
                args += (pixel_type,)
                if fallback_only is not None:
                    args += (fallback_only,)
            self._img = _read_function(self.request.get_local_filename(), *args)

        def _get_length(self):
            return 1

        def _close(self):
            pass

        def _get_data(self, index):
            # Get data
            if index != 0:
                error_msg = "Index out of range while reading from itk file"
                raise IndexError(error_msg)

            # Return array and empty meta data
            return _itk.GetArrayFromImage(self._img), {}

        def _get_meta_data(self, index):
            error_msg = "The itk plugin does not support meta data, currently."
            raise RuntimeError(error_msg)

    # -- writer
    class Writer(Format.Writer):
        def _open(self):
            if not _itk:
                load_lib()

        def _close(self):
            pass

        def _append_data(self, im, meta):
            _itk_img = _itk.GetImageFromArray(im)
            _write_function(_itk_img, self.request.get_local_filename())

        def set_meta_data(self, meta):
            error_msg = "The itk plugin does not support meta data, currently."
            raise RuntimeError(error_msg)
