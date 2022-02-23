# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Read SPE files.

Backend: internal

This plugin supports reading files saved in the Princeton Instruments
SPE file format.

Parameters for reading
----------------------
char_encoding : str
    Character encoding used to decode strings in the metadata. Defaults
    to "latin1".
check_filesize : bool
    The number of frames in the file is stored in the file header. However,
    this number may be wrong for certain software. If this is `True`
    (default), derive the number of frames also from the file size and
    raise a warning if the two values do not match.
sdt_meta : bool
    If set to `True` (default), check for special metadata written by the
    `SDT-control` software. Does not have an effect for files written by
    other software.

Metadata for reading
--------------------
ROIs : list of dict
    Regions of interest used for recording images. Each dict has the
    "top_left" key containing x and y coordinates of the top left corner,
    the "bottom_right" key with x and y coordinates of the bottom right
    corner, and the "bin" key with number of binned pixels in x and y
    directions.
comments : list of str
    The SPE format allows for 5 comment strings of 80 characters each.
controller_version : int
    Hardware version
logic_output : int
    Definition of output BNC
amp_hi_cap_low_noise : int
    Amp switching mode
mode : int
    Timing mode
exp_sec : float
    Alternative exposure in seconds
date : str
    Date string
detector_temp : float
    Detector temperature
detector_type : int
    CCD / diode array type
st_diode : int
    Trigger diode
delay_time : float
    Used with async mode
shutter_control : int
    Normal, disabled open, or disabled closed
absorb_live : bool
    on / off
absorb_mode : int
    Reference strip or file
can_do_virtual_chip : bool
    True or False whether chip can do virtual chip
threshold_min_live : bool
    on / off
threshold_min_val : float
    Threshold minimum value
threshold_max_live : bool
    on / off
threshold_max_val : float
    Threshold maximum value
time_local : str
    Experiment local time
time_utc : str
    Experiment UTC time
adc_offset : int
    ADC offset
adc_rate : int
    ADC rate
adc_type : int
    ADC type
adc_resolution : int
    ADC resolution
adc_bit_adjust : int
    ADC bit adjust
gain : int
    gain
sw_version : str
    Version of software which created this file
spare_4 : bytes
    Reserved space
readout_time : float
    Experiment readout time
type : str
    Controller type
clockspeed_us : float
    Vertical clock speed in microseconds
readout_mode : ["full frame", "frame transfer", "kinetics", ""]
    Readout mode. Empty string means that this was not set by the
    Software.
window_size : int
    Window size for Kinetics mode
file_header_ver : float
    File header version
chip_size : [int, int]
    x and y dimensions of the camera chip
virt_chip_size : [int, int]
    Virtual chip x and y dimensions
pre_pixels : [int, int]
    Pre pixels in x and y dimensions
post_pixels : [int, int],
    Post pixels in x and y dimensions
geometric : list of {"rotate", "reverse", "flip"}
    Geometric operations
sdt_major_version : int
    (only for files created by SDT-control)
    Major version of SDT-control software
sdt_minor_version : int
    (only for files created by SDT-control)
    Minor version of SDT-control software
sdt_controller_name : str
    (only for files created by SDT-control)
    Controller name
exposure_time : float
    (only for files created by SDT-control)
    Exposure time in seconds
color_code : str
    (only for files created by SDT-control)
    Color channels used
detection_channels : int
    (only for files created by SDT-control)
    Number of channels
background_subtraction : bool
    (only for files created by SDT-control)
    Whether background subtraction war turned on
em_active : bool
    (only for files created by SDT-control)
    Whether EM was turned on
em_gain : int
    (only for files created by SDT-control)
    EM gain
modulation_active : bool
    (only for files created by SDT-control)
    Whether laser modulation (“attenuate”) was turned on
pixel_size : float
    (only for files created by SDT-control)
    Camera pixel size
sequence_type : str
    (only for files created by SDT-control)
    Type of sequnce (standard, TOCCSL, arbitrary, …)
grid : float
    (only for files created by SDT-control)
    Sequence time unit (“grid size”) in seconds
n_macro : int
    (only for files created by SDT-control)
    Number of macro loops
delay_macro : float
    (only for files created by SDT-control)
    Time between macro loops in seconds
n_mini : int
    (only for files created by SDT-control)
    Number of mini loops
delay_mini : float
    (only for files created by SDT-control)
    Time between mini loops in seconds
n_micro : int (only for files created by SDT-control)
    Number of micro loops
delay_micro : float (only for files created by SDT-control)
    Time between micro loops in seconds
n_subpics : int
    (only for files created by SDT-control)
    Number of sub-pictures
delay_shutter : float
    (only for files created by SDT-control)
    Camera shutter delay in seconds
delay_prebleach : float
    (only for files created by SDT-control)
    Pre-bleach delay in seconds
bleach_time : float
    (only for files created by SDT-control)
    Bleaching time in seconds
recovery_time : float
    (only for files created by SDT-control)
    Recovery time in seconds
comment : str
    (only for files created by SDT-control)
    User-entered comment. This replaces the "comments" field.
datetime : datetime.datetime
    (only for files created by SDT-control)
    Combines the "date" and "time_local" keys. The latter two plus
    "time_utc" are removed.
modulation_script : str
    (only for files created by SDT-control)
    Laser modulation script. Replaces the "spare_4" key.

"""

from datetime import datetime
import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from ..core import Format


logger = logging.getLogger(__name__)


class Spec:
    """SPE file specification data

    Tuples of (offset, datatype, count), where offset is the offset in the SPE
    file and datatype is the datatype as used in `numpy.fromfile`()

    `data_start` is the offset of actual image data.

    `dtypes` translates SPE datatypes (0...4) to numpy ones, e. g. dtypes[0]
    is dtype("<f") (which is np.float32).

    `controllers` maps the `type` metadata to a human readable name

    `readout_modes` maps the `readoutMode` metadata to something human readable
    although this may not be accurate since there is next to no documentation
    to be found.
    """

    basic = {
        "datatype": (108, "<h"),  # dtypes
        "xdim": (42, "<H"),
        "ydim": (656, "<H"),
        "xml_footer_offset": (678, "<Q"),
        "NumFrames": (1446, "<i"),
        "file_header_ver": (1992, "<f"),
    }

    metadata = {
        # ROI information
        "NumROI": (1510, "<h"),
        "ROIs": (
            1512,
            np.dtype(
                [
                    ("startx", "<H"),
                    ("endx", "<H"),
                    ("groupx", "<H"),
                    ("starty", "<H"),
                    ("endy", "<H"),
                    ("groupy", "<H"),
                ]
            ),
            10,
        ),
        # chip-related sizes
        "xDimDet": (6, "<H"),
        "yDimDet": (18, "<H"),
        "VChipXdim": (14, "<h"),
        "VChipYdim": (16, "<h"),
        # other stuff
        "controller_version": (0, "<h"),
        "logic_output": (2, "<h"),
        "amp_high_cap_low_noise": (4, "<H"),  # enum?
        "mode": (8, "<h"),  # enum?
        "exposure_sec": (10, "<f"),
        "date": (20, "<10S"),
        "detector_temp": (36, "<f"),
        "detector_type": (40, "<h"),
        "st_diode": (44, "<h"),
        "delay_time": (46, "<f"),
        # shutter_control: normal, disabled open, disabled closed
        # But which one is which?
        "shutter_control": (50, "<H"),
        "absorb_live": (52, "<h"),
        "absorb_mode": (54, "<H"),
        "can_do_virtual_chip": (56, "<h"),
        "threshold_min_live": (58, "<h"),
        "threshold_min_val": (60, "<f"),
        "threshold_max_live": (64, "<h"),
        "threshold_max_val": (66, "<f"),
        "time_local": (172, "<7S"),
        "time_utc": (179, "<7S"),
        "adc_offset": (188, "<H"),
        "adc_rate": (190, "<H"),
        "adc_type": (192, "<H"),
        "adc_resolution": (194, "<H"),
        "adc_bit_adjust": (196, "<H"),
        "gain": (198, "<H"),
        "comments": (200, "<80S", 5),
        "geometric": (600, "<H"),  # flags
        "sw_version": (688, "<16S"),
        "spare_4": (742, "<436S"),
        "XPrePixels": (98, "<h"),
        "XPostPixels": (100, "<h"),
        "YPrePixels": (102, "<h"),
        "YPostPixels": (104, "<h"),
        "readout_time": (672, "<f"),
        "xml_footer_offset": (678, "<Q"),
        "type": (704, "<h"),  # controllers
        "clockspeed_us": (1428, "<f"),
        "readout_mode": (1480, "<H"),  # readout_modes
        "window_size": (1482, "<H"),
        "file_header_ver": (1992, "<f"),
    }

    data_start = 4100

    dtypes = {
        0: np.dtype(np.float32),
        1: np.dtype(np.int32),
        2: np.dtype(np.int16),
        3: np.dtype(np.uint16),
        8: np.dtype(np.uint32),
    }

    controllers = [
        "new120 (Type II)",
        "old120 (Type I)",
        "ST130",
        "ST121",
        "ST138",
        "DC131 (PentaMax)",
        "ST133 (MicroMax/Roper)",
        "ST135 (GPIB)",
        "VTCCD",
        "ST116 (GPIB)",
        "OMA3 (GPIB)",
        "OMA4",
    ]

    # This was gathered from random places on the internet and own experiments
    # with the camera. May not be accurate.
    readout_modes = ["full frame", "frame transfer", "kinetics"]

    # Do not decode the following metadata keys into strings, but leave them
    # as byte arrays
    no_decode = ["spare_4"]


class SDTControlSpec:
    """Extract metadata written by the SDT-control software

    Some of it is encoded in the comment strings
    (see :py:meth:`parse_comments`). Also, date and time are encoded in a
    peculiar way (see :py:meth:`get_datetime`). Use :py:meth:`extract_metadata`
    to update the metadata dict.
    """

    months = {
        # Convert SDT-control month strings to month numbers
        "Jän": 1,
        "Jan": 1,
        "Feb": 2,
        "Mär": 3,
        "Mar": 3,
        "Apr": 4,
        "Mai": 5,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Okt": 10,
        "Oct": 10,
        "Nov": 11,
        "Dez": 12,
        "Dec": 12,
    }

    sequence_types = {
        # TODO: complete
        "SEQU": "standard",
        "SETO": "TOCCSL",
        "KINE": "kinetics",
        "SEAR": "arbitrary",
    }

    class CommentDesc:
        """Describe how to extract a metadata entry from a comment string"""

        n: int
        """Which of the 5 SPE comment fields to use."""
        slice: slice
        """Which characters from the `n`-th comment to use."""
        cvt: Callable[[str], Any]
        """How to convert characters to something useful."""
        scale: Union[None, float]
        """Optional scaling factor for numbers"""

        def __init__(
            self,
            n: int,
            slice: slice,
            cvt: Callable[[str], Any] = str,
            scale: Optional[float] = None,
        ):
            self.n = n
            self.slice = slice
            self.cvt = cvt
            self.scale = scale

    comments = {
        "sdt_major_version": CommentDesc(4, slice(66, 68), int),
        "sdt_minor_version": CommentDesc(4, slice(68, 70), int),
        "sdt_controller_name": CommentDesc(4, slice(0, 6), str),
        "exposure_time": CommentDesc(1, slice(64, 73), float, 10 ** -6),
        "color_code": CommentDesc(4, slice(10, 14), str),
        "detection_channels": CommentDesc(4, slice(15, 16), int),
        "background_subtraction": CommentDesc(4, 14, lambda x: x == "B"),
        "em_active": CommentDesc(4, 32, lambda x: x == "E"),
        "em_gain": CommentDesc(4, slice(28, 32), int),
        "modulation_active": CommentDesc(4, 33, lambda x: x == "A"),
        "pixel_size": CommentDesc(4, slice(25, 28), float, 0.1),
        "sequence_type": CommentDesc(
            4, slice(6, 10), lambda x: __class__.sequence_types[x]
        ),
        "grid": CommentDesc(4, slice(16, 25), float, 10 ** -6),
        "n_macro": CommentDesc(1, slice(0, 4), int),
        "delay_macro": CommentDesc(1, slice(10, 19), float, 10 ** -3),
        "n_mini": CommentDesc(1, slice(4, 7), int),
        "delay_mini": CommentDesc(1, slice(19, 28), float, 10 ** -6),
        "n_micro": CommentDesc(1, slice(7, 10), int),
        "delay_micro": CommentDesc(1, slice(28, 37), float, 10 ** -6),
        "n_subpics": CommentDesc(1, slice(7, 10), int),
        "delay_shutter": CommentDesc(1, slice(73, 79), float, 10 ** -6),
        "delay_prebleach": CommentDesc(1, slice(37, 46), float, 10 ** -6),
        "bleach_time": CommentDesc(1, slice(46, 55), float, 10 ** -6),
        "recovery_time": CommentDesc(1, slice(55, 64), float, 10 ** -6),
    }

    @staticmethod
    def parse_comments(comments: Sequence[str]) -> Union[Dict, None]:
        """Extract SDT-control metadata from comments

        Parameters
        ----------
        comments
            List of SPE file comments, typically ``metadata["comments"]``.

        Returns
        -------
        If SDT-control comments were detected, return a dict of metadata, else
        `None`.
        """
        sdt_md = {}
        if comments[4][70:] != "COMVER0500":
            logger.debug("SDT-control comments not found.")
            return None

        sdt_md = {}
        for name, spec in SDTControlSpec.comments.items():
            try:
                v = spec.cvt(comments[spec.n][spec.slice])
                if spec.scale is not None:
                    v *= spec.scale
            except Exception as e:
                logger.debug(
                    "Failed to decode SDT-control metadata " f'field "{name}": {e}'
                )
            sdt_md[name] = v
        comment = comments[0] + comments[2]
        sdt_md["comment"] = comment.strip()
        return sdt_md

    @staticmethod
    def get_datetime(date: str, time: str) -> Union[datetime, None]:
        """Turn date and time saved by SDT-control into proper datetime object

        Parameters
        ----------
        date
            SPE file date, typically ``metadata["date"]``.
        time
            SPE file date, typically ``metadata["time_local"]``.

        Returns
        -------
        File's datetime if parsing was succsessful, else None.
        """
        try:
            month = __class__.months[date[2:5]]
            return datetime(
                int(date[5:9]),
                month,
                int(date[0:2]),
                int(time[0:2]),
                int(time[2:4]),
                int(time[4:6]),
            )
        except Exception as e:
            logger.info(f"Failed to decode date from SDT-control metadata: {e}.")

    @staticmethod
    def extract_metadata(meta: Mapping, char_encoding: str = "latin1"):
        """Extract SDT-control metadata from SPE metadata

        SDT-control stores some metadata in comments and other fields.
        Extract them and remove unused entries.

        Parameters
        ----------
        meta
            SPE file metadata. Modified in place.
        char_encoding
            Character encoding used to decode strings in the metadata.
        """
        sdt_meta = __class__.parse_comments(meta["comments"])
        if not sdt_meta:
            return
        # This file has SDT-control metadata
        meta.pop("comments")
        meta.update(sdt_meta)

        # Get date and time in a usable format
        dt = __class__.get_datetime(meta["date"], meta["time_local"])
        if dt:
            meta["datetime"] = dt
            meta.pop("date")
            meta.pop("time_local")

        sp4 = meta["spare_4"]
        try:
            meta["modulation_script"] = sp4.decode(char_encoding)
            meta.pop("spare_4")
        except UnicodeDecodeError:
            logger.warning(
                "Failed to decode SDT-control laser "
                "modulation script. Bad char_encoding?"
            )

        # Get rid of unused data
        meta.pop("time_utc")
        meta.pop("exposure_sec")


class SpeFormat(Format):
    """See :mod:`imageio.plugins.spe`"""

    def _can_read(self, request):
        return (
            request.mode[1] in self.modes + "?" and request.extension in self.extensions
        )

    def _can_write(self, request):
        return False

    class Reader(Format.Reader):
        def _open(self, char_encoding="latin1", check_filesize=True, sdt_meta=True):
            self._file = self.request.get_file()
            self._char_encoding = char_encoding

            info = self._parse_header(Spec.basic)
            self._file_header_ver = info["file_header_ver"]
            self._dtype = Spec.dtypes[info["datatype"]]
            self._shape = (info["ydim"], info["xdim"])
            self._len = info["NumFrames"]
            self._sdt_meta = sdt_meta

            if check_filesize:
                # Some software writes incorrect `NumFrames` metadata.
                # To determine the number of frames, check the size of the data
                # segment -- until the end of the file for SPE<3, until the
                # xml footer for SPE>=3.
                data_end = (
                    info["xml_footer_offset"]
                    if info["file_header_ver"] >= 3
                    else os.path.getsize(self.request.get_local_filename())
                )
                line = data_end - Spec.data_start
                line //= self._shape[0] * self._shape[1] * self._dtype.itemsize
                if line != self._len:
                    logger.warning(
                        "The file header of %s claims there are %s frames, "
                        "but there are actually %s frames.",
                        self.request.filename,
                        self._len,
                        line,
                    )
                    self._len = min(line, self._len)

            self._meta = None

        def _get_meta_data(self, index):
            if self._meta is None:
                if self._file_header_ver < 3:
                    self._init_meta_data_pre_v3()
                else:
                    self._init_meta_data_post_v3()
            return self._meta

        def _close(self):
            # The file should be closed by `self.request`
            pass

        def _init_meta_data_pre_v3(self):
            self._meta = self._parse_header(Spec.metadata)

            nr = self._meta.pop("NumROI", None)
            nr = 1 if nr < 1 else nr
            self._meta["ROIs"] = roi_array_to_dict(self._meta["ROIs"][:nr])

            # chip sizes
            self._meta["chip_size"] = [
                self._meta.pop("xDimDet", None),
                self._meta.pop("yDimDet", None),
            ]
            self._meta["virt_chip_size"] = [
                self._meta.pop("VChipXdim", None),
                self._meta.pop("VChipYdim", None),
            ]
            self._meta["pre_pixels"] = [
                self._meta.pop("XPrePixels", None),
                self._meta.pop("YPrePixels", None),
            ]
            self._meta["post_pixels"] = [
                self._meta.pop("XPostPixels", None),
                self._meta.pop("YPostPixels", None),
            ]

            # comments
            self._meta["comments"] = [str(c) for c in self._meta["comments"]]

            # geometric operations
            g = []
            f = self._meta.pop("geometric", 0)
            if f & 1:
                g.append("rotate")
            if f & 2:
                g.append("reverse")
            if f & 4:
                g.append("flip")
            self._meta["geometric"] = g

            # Make some additional information more human-readable
            t = self._meta["type"]
            if 1 <= t <= len(Spec.controllers):
                self._meta["type"] = Spec.controllers[t - 1]
            else:
                self._meta["type"] = ""
            m = self._meta["readout_mode"]
            if 1 <= m <= len(Spec.readout_modes):
                self._meta["readout_mode"] = Spec.readout_modes[m - 1]
            else:
                self._meta["readout_mode"] = ""

            # bools
            for k in (
                "absorb_live",
                "can_do_virtual_chip",
                "threshold_min_live",
                "threshold_max_live",
            ):
                self._meta[k] = bool(self._meta[k])

            # frame shape
            self._meta["frame_shape"] = self._shape

            # Extract SDT-control metadata if desired
            if self._sdt_meta:
                SDTControlSpec.extract_metadata(self._meta, self._char_encoding)

        def _parse_header(self, spec):
            ret = {}
            # Decode each string from the numpy array read by np.fromfile
            decode = np.vectorize(lambda x: x.decode(self._char_encoding))

            for name, sp in spec.items():
                self._file.seek(sp[0])
                cnt = 1 if len(sp) < 3 else sp[2]
                v = np.fromfile(self._file, dtype=sp[1], count=cnt)
                if v.dtype.kind == "S" and name not in Spec.no_decode:
                    # Silently ignore string decoding failures
                    try:
                        v = decode(v)
                    except Exception:
                        logger.warning(
                            'Failed to decode "{}" metadata '
                            "string. Check `char_encoding` "
                            "parameter.".format(name)
                        )

                try:
                    # For convenience, if the array contains only one single
                    # entry, return this entry itself.
                    v = v.item()
                except ValueError:
                    v = np.squeeze(v)
                ret[name] = v
            return ret

        def _init_meta_data_post_v3(self):
            info = self._parse_header(Spec.basic)
            self._file.seek(info["xml_footer_offset"])
            xml = self._file.read()
            self._meta = {"__xml": xml}

        def _get_length(self):
            if self.request.mode[1] in "vV":
                return 1
            else:
                return self._len

        def _get_data(self, index):
            if index < 0:
                raise IndexError("Image index %i < 0" % index)
            if index >= self._len:
                raise IndexError("Image index %i > %i" % (index, self._len))

            if self.request.mode[1] in "vV":
                if index != 0:
                    raise IndexError("Index has to be 0 in v and V modes")
                self._file.seek(Spec.data_start)
                data = np.fromfile(
                    self._file,
                    dtype=self._dtype,
                    count=self._shape[0] * self._shape[1] * self._len,
                )
                data = data.reshape((self._len,) + self._shape)
            else:
                self._file.seek(
                    Spec.data_start
                    + index * self._shape[0] * self._shape[1] * self._dtype.itemsize
                )
                data = np.fromfile(
                    self._file, dtype=self._dtype, count=self._shape[0] * self._shape[1]
                )
                data = data.reshape(self._shape)
            return data, self._get_meta_data(index)


def roi_array_to_dict(a):
    """Convert the `ROIs` structured arrays to :py:class:`dict`

    Parameters
    ----------
    a : numpy.ndarray:
        Structured array containing ROI data

    Returns
    -------
    list of dict
        One dict per ROI. Keys are "top_left", "bottom_right", and "bin",
        values are tuples whose first element is the x axis value and the
        second element is the y axis value.
    """
    dict_list = []
    a = a[["startx", "starty", "endx", "endy", "groupx", "groupy"]]
    for sx, sy, ex, ey, gx, gy in a:
        roi_dict = {
            "top_left": [int(sx), int(sy)],
            "bottom_right": [int(ex), int(ey)],
            "bin": [int(gx), int(gy)],
        }
        dict_list.append(roi_dict)
    return dict_list
