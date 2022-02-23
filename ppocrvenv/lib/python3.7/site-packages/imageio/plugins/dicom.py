# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read DICOM files.

Backend Library: internal

A format for reading DICOM images: a common format used to store
medical image data, such as X-ray, CT and MRI.

This format borrows some code (and ideas) from the pydicom project. However,
only a predefined subset of tags are extracted from the file. This allows
for great simplifications allowing us to make a stand-alone reader, and
also results in a much faster read time.

By default, only uncompressed and deflated transfer syntaxes are supported.
If gdcm or dcmtk is installed, these will be used to automatically convert
the data. See https://github.com/malaterre/GDCM/releases for installing GDCM.

This format provides functionality to group images of the same
series together, thus extracting volumes (and multiple volumes).
Using volread will attempt to yield a volume. If multiple volumes
are present, the first one is given. Using mimread will simply yield
all images in the given directory (not taking series into account).

Parameters
----------
progress : {True, False, BaseProgressIndicator}
    Whether to show progress when reading from multiple files.
    Default True. By passing an object that inherits from
    BaseProgressIndicator, the way in which progress is reported
    can be costumized.

"""

# todo: Use pydicom:
# * Note: is not py3k ready yet
# * Allow reading the full meta info
# I think we can more or less replace the SimpleDicomReader with a
# pydicom.Dataset For series, only ned to read the full info from one
# file: speed still high
# * Perhaps allow writing?

import os
import sys
import logging
import subprocess

from ..core import Format, BaseProgressIndicator, StdoutProgressIndicator
from ..core import read_n_bytes

_dicom = None  # lazily loaded in load_lib()

logger = logging.getLogger(__name__)


def load_lib():
    global _dicom
    from . import _dicom

    return _dicom


# Determine endianity of system
sys_is_little_endian = sys.byteorder == "little"


def get_dcmdjpeg_exe():
    fname = "dcmdjpeg" + ".exe" * sys.platform.startswith("win")
    for dir in (
        "c:\\dcmtk",
        "c:\\Program Files",
        "c:\\Program Files\\dcmtk",
        "c:\\Program Files (x86)\\dcmtk",
    ):
        filename = os.path.join(dir, fname)
        if os.path.isfile(filename):
            return [filename]

    try:
        subprocess.check_call([fname, "--version"])
        return [fname]
    except Exception:
        return None


def get_gdcmconv_exe():
    fname = "gdcmconv" + ".exe" * sys.platform.startswith("win")
    # Maybe it's on the path
    try:
        subprocess.check_call([fname, "--version"])
        return [fname, "--raw"]
    except Exception:
        pass
    # Select directories where it could be
    candidates = []
    base_dirs = [r"c:\Program Files"]
    for base_dir in base_dirs:
        if os.path.isdir(base_dir):
            for dname in os.listdir(base_dir):
                if dname.lower().startswith("gdcm"):
                    suffix = dname[4:].strip()
                    candidates.append((suffix, os.path.join(base_dir, dname)))
    # Sort, so higher versions are tried earlier
    candidates.sort(reverse=True)
    # Select executable
    filename = None
    for _, dirname in candidates:
        exe1 = os.path.join(dirname, "gdcmconv.exe")
        exe2 = os.path.join(dirname, "bin", "gdcmconv.exe")
        if os.path.isfile(exe1):
            filename = exe1
            break
        if os.path.isfile(exe2):
            filename = exe2
            break
    else:
        return None
    return [filename, "--raw"]


class DicomFormat(Format):
    """See :mod:`imageio.plugins.dicom`"""

    def _can_read(self, request):
        # If user URI was a directory, we check whether it has a DICOM file
        if os.path.isdir(request.filename):
            files = os.listdir(request.filename)
            for fname in sorted(files):  # Sorting make it consistent
                filename = os.path.join(request.filename, fname)
                if os.path.isfile(filename) and "DICOMDIR" not in fname:
                    with open(filename, "rb") as f:
                        first_bytes = read_n_bytes(f, 140)
                    return first_bytes[128:132] == b"DICM"
            else:
                return False
        # Check
        return request.firstbytes[128:132] == b"DICM"

    def _can_write(self, request):
        # We cannot save yet. May be possible if we will used pydicom as
        # a backend.
        return False

    # --

    class Reader(Format.Reader):

        _compressed_warning_dirs = set()

        def _open(self, progress=True):
            if not _dicom:
                load_lib()
            if os.path.isdir(self.request.filename):
                # A dir can be given if the user used the format explicitly
                self._info = {}
                self._data = None
            else:
                # Read the given dataset now ...
                try:
                    dcm = _dicom.SimpleDicomReader(self.request.get_file())
                except _dicom.CompressedDicom as err:
                    # We cannot do this on our own. Perhaps with some help ...
                    cmd = get_gdcmconv_exe()
                    if not cmd and "JPEG" in str(err):
                        cmd = get_dcmdjpeg_exe()
                    if not cmd:
                        msg = err.args[0].replace("using", "installing")
                        msg = msg.replace("convert", "auto-convert")
                        err.args = (msg,)
                        raise
                    else:
                        fname1 = self.request.get_local_filename()
                        fname2 = fname1 + ".raw"
                        try:
                            subprocess.check_call(cmd + [fname1, fname2])
                        except Exception:
                            raise err
                        d = os.path.dirname(fname1)
                        if d not in self._compressed_warning_dirs:
                            self._compressed_warning_dirs.add(d)
                            logger.warning(
                                "DICOM file contained compressed data. "
                                + "Autoconverting with "
                                + cmd[0]
                                + " (this warning is shown once for each directory)"
                            )
                        dcm = _dicom.SimpleDicomReader(fname2)

                self._info = dcm._info
                self._data = dcm.get_numpy_array()

            # Initialize series, list of DicomSeries objects
            self._series = None  # only created if needed

            # Set progress indicator
            if isinstance(progress, BaseProgressIndicator):
                self._progressIndicator = progress
            elif progress is True:
                p = StdoutProgressIndicator("Reading DICOM")
                self._progressIndicator = p
            elif progress in (None, False):
                self._progressIndicator = BaseProgressIndicator("Dummy")
            else:
                raise ValueError("Invalid value for progress.")

        def _close(self):
            # Clean up
            self._info = None
            self._data = None
            self._series = None

        @property
        def series(self):
            if self._series is None:
                pi = self._progressIndicator
                self._series = _dicom.process_directory(self.request, pi)
            return self._series

        def _get_length(self):
            if self._data is None:
                dcm = self.series[0][0]
                self._info = dcm._info
                self._data = dcm.get_numpy_array()

            nslices = self._data.shape[0] if (self._data.ndim == 3) else 1

            if self.request.mode[1] == "i":
                # User expects one, but lets be honest about this file
                return nslices
            elif self.request.mode[1] == "I":
                # User expects multiple, if this file has multiple slices, ok.
                # Otherwise we have to check the series.
                if nslices > 1:
                    return nslices
                else:
                    return sum([len(serie) for serie in self.series])
            elif self.request.mode[1] == "v":
                # User expects a volume, if this file has one, ok.
                # Otherwise we have to check the series
                if nslices > 1:
                    return 1
                else:
                    return len(self.series)  # We assume one volume per series
            elif self.request.mode[1] == "V":
                # User expects multiple volumes. We have to check the series
                return len(self.series)  # We assume one volume per series
            else:
                raise RuntimeError("DICOM plugin should know what to expect.")

        def _get_data(self, index):
            if self._data is None:
                dcm = self.series[0][0]
                self._info = dcm._info
                self._data = dcm.get_numpy_array()

            nslices = self._data.shape[0] if (self._data.ndim == 3) else 1

            if self.request.mode[1] == "i":
                # Allow index >1 only if this file contains >1
                if nslices > 1:
                    return self._data[index], self._info
                elif index == 0:
                    return self._data, self._info
                else:
                    raise IndexError("Dicom file contains only one slice.")
            elif self.request.mode[1] == "I":
                # Return slice from volume, or return item from series
                if index == 0 and nslices > 1:
                    return self._data[index], self._info
                else:
                    L = []
                    for serie in self.series:
                        L.extend([dcm_ for dcm_ in serie])
                    return L[index].get_numpy_array(), L[index].info
            elif self.request.mode[1] in "vV":
                # Return volume or series
                if index == 0 and nslices > 1:
                    return self._data, self._info
                else:
                    return (
                        self.series[index].get_numpy_array(),
                        self.series[index].info,
                    )
            else:  # pragma: no cover
                raise ValueError("DICOM plugin should know what to expect.")

        def _get_meta_data(self, index):
            if self._data is None:
                dcm = self.series[0][0]
                self._info = dcm._info
                self._data = dcm.get_numpy_array()

            nslices = self._data.shape[0] if (self._data.ndim == 3) else 1

            # Default is the meta data of the given file, or the "first" file.
            if index is None:
                return self._info

            if self.request.mode[1] == "i":
                return self._info
            elif self.request.mode[1] == "I":
                # Return slice from volume, or return item from series
                if index == 0 and nslices > 1:
                    return self._info
                else:
                    L = []
                    for serie in self.series:
                        L.extend([dcm_ for dcm_ in serie])
                    return L[index].info
            elif self.request.mode[1] in "vV":
                # Return volume or series
                if index == 0 and nslices > 1:
                    return self._info
                else:
                    return self.series[index].info
            else:  # pragma: no cover
                raise ValueError("DICOM plugin should know what to expect.")
