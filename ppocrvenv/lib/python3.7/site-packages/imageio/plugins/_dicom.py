# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

""" Plugin for reading DICOM files.
"""

# todo: Use pydicom:
# * Note: is not py3k ready yet
# * Allow reading the full meta info
# I think we can more or less replace the SimpleDicomReader with a
# pydicom.Dataset For series, only ned to read the full info from one
# file: speed still high
# * Perhaps allow writing?

import sys
import os
import struct
import logging

import numpy as np


logger = logging.getLogger(__name__)

# Determine endianity of system
sys_is_little_endian = sys.byteorder == "little"

# Define a dictionary that contains the tags that we would like to know
MINIDICT = {
    (0x7FE0, 0x0010): ("PixelData", "OB"),
    # Date and time
    (0x0008, 0x0020): ("StudyDate", "DA"),
    (0x0008, 0x0021): ("SeriesDate", "DA"),
    (0x0008, 0x0022): ("AcquisitionDate", "DA"),
    (0x0008, 0x0023): ("ContentDate", "DA"),
    (0x0008, 0x0030): ("StudyTime", "TM"),
    (0x0008, 0x0031): ("SeriesTime", "TM"),
    (0x0008, 0x0032): ("AcquisitionTime", "TM"),
    (0x0008, 0x0033): ("ContentTime", "TM"),
    # With what, where, by whom?
    (0x0008, 0x0060): ("Modality", "CS"),
    (0x0008, 0x0070): ("Manufacturer", "LO"),
    (0x0008, 0x0080): ("InstitutionName", "LO"),
    # Descriptions
    (0x0008, 0x1030): ("StudyDescription", "LO"),
    (0x0008, 0x103E): ("SeriesDescription", "LO"),
    # UID's
    (0x0008, 0x0016): ("SOPClassUID", "UI"),
    (0x0008, 0x0018): ("SOPInstanceUID", "UI"),
    (0x0020, 0x000D): ("StudyInstanceUID", "UI"),
    (0x0020, 0x000E): ("SeriesInstanceUID", "UI"),
    (0x0008, 0x0117): ("ContextUID", "UI"),
    # Numbers
    (0x0020, 0x0011): ("SeriesNumber", "IS"),
    (0x0020, 0x0012): ("AcquisitionNumber", "IS"),
    (0x0020, 0x0013): ("InstanceNumber", "IS"),
    (0x0020, 0x0014): ("IsotopeNumber", "IS"),
    (0x0020, 0x0015): ("PhaseNumber", "IS"),
    (0x0020, 0x0016): ("IntervalNumber", "IS"),
    (0x0020, 0x0017): ("TimeSlotNumber", "IS"),
    (0x0020, 0x0018): ("AngleNumber", "IS"),
    (0x0020, 0x0019): ("ItemNumber", "IS"),
    (0x0020, 0x0020): ("PatientOrientation", "CS"),
    (0x0020, 0x0030): ("ImagePosition", "CS"),
    (0x0020, 0x0032): ("ImagePositionPatient", "CS"),
    (0x0020, 0x0035): ("ImageOrientation", "CS"),
    (0x0020, 0x0037): ("ImageOrientationPatient", "CS"),
    # Patient information
    (0x0010, 0x0010): ("PatientName", "PN"),
    (0x0010, 0x0020): ("PatientID", "LO"),
    (0x0010, 0x0030): ("PatientBirthDate", "DA"),
    (0x0010, 0x0040): ("PatientSex", "CS"),
    (0x0010, 0x1010): ("PatientAge", "AS"),
    (0x0010, 0x1020): ("PatientSize", "DS"),
    (0x0010, 0x1030): ("PatientWeight", "DS"),
    # Image specific (required to construct numpy array)
    (0x0028, 0x0002): ("SamplesPerPixel", "US"),
    (0x0028, 0x0008): ("NumberOfFrames", "IS"),
    (0x0028, 0x0100): ("BitsAllocated", "US"),
    (0x0028, 0x0101): ("BitsStored", "US"),
    (0x0028, 0x0102): ("HighBit", "US"),
    (0x0028, 0x0103): ("PixelRepresentation", "US"),
    (0x0028, 0x0010): ("Rows", "US"),
    (0x0028, 0x0011): ("Columns", "US"),
    (0x0028, 0x1052): ("RescaleIntercept", "DS"),
    (0x0028, 0x1053): ("RescaleSlope", "DS"),
    # Image specific (for the user)
    (0x0028, 0x0030): ("PixelSpacing", "DS"),
    (0x0018, 0x0088): ("SliceSpacing", "DS"),
}

# Define some special tags:
# See PS 3.5-2008 section 7.5 (p.40)
ItemTag = (0xFFFE, 0xE000)  # start of Sequence Item
ItemDelimiterTag = (0xFFFE, 0xE00D)  # end of Sequence Item
SequenceDelimiterTag = (0xFFFE, 0xE0DD)  # end of Sequence of undefined length

# Define set of groups that we're interested in (so we can quickly skip others)
GROUPS = set([key[0] for key in MINIDICT.keys()])
VRS = set([val[1] for val in MINIDICT.values()])


class NotADicomFile(Exception):
    pass


class CompressedDicom(RuntimeError):
    pass


class SimpleDicomReader(object):
    """
    This class provides reading of pixel data from DICOM files. It is
    focussed on getting the pixel data, not the meta info.

    To use, first create an instance of this class (giving it
    a file object or filename). Next use the info attribute to
    get a dict of the meta data. The loading of pixel data is
    deferred until get_numpy_array() is called.

    Comparison with Pydicom
    -----------------------

    This code focusses on getting the pixel data out, which allows some
    shortcuts, resulting in the code being much smaller.

    Since the processing of data elements is much cheaper (it skips a lot
    of tags), this code is about 3x faster than pydicom (except for the
    deflated DICOM files).

    This class does borrow some code (and ideas) from the pydicom
    project, and (to the best of our knowledge) has the same limitations
    as pydicom with regard to the type of files that it can handle.

    Limitations
    -----------

    For more advanced DICOM processing, please check out pydicom.

      * Only a predefined subset of data elements (meta information) is read.
      * This is a reader; it can not write DICOM files.
      * (just like pydicom) it can handle none of the compressed DICOM
        formats except for "Deflated Explicit VR Little Endian"
        (1.2.840.10008.1.2.1.99).

    """

    def __init__(self, file):
        # Open file if filename given
        if isinstance(file, str):
            self._filename = file
            self._file = open(file, "rb")
        else:
            self._filename = "<unknown file>"
            self._file = file
        # Init variable to store position and size of pixel data
        self._pixel_data_loc = None
        # The meta header is always explicit and little endian
        self.is_implicit_VR = False
        self.is_little_endian = True
        self._unpackPrefix = "<"
        # Dict to store data elements of interest in
        self._info = {}
        # VR Conversion
        self._converters = {
            # Numbers
            "US": lambda x: self._unpack("H", x),
            "UL": lambda x: self._unpack("L", x),
            # Numbers encoded as strings
            "DS": lambda x: self._splitValues(x, float, "\\"),
            "IS": lambda x: self._splitValues(x, int, "\\"),
            # strings
            "AS": lambda x: x.decode("ascii", "ignore").strip("\x00"),
            "DA": lambda x: x.decode("ascii", "ignore").strip("\x00"),
            "TM": lambda x: x.decode("ascii", "ignore").strip("\x00"),
            "UI": lambda x: x.decode("ascii", "ignore").strip("\x00"),
            "LO": lambda x: x.decode("utf-8", "ignore").strip("\x00").rstrip(),
            "CS": lambda x: self._splitValues(x, float, "\\"),
            "PN": lambda x: x.decode("utf-8", "ignore").strip("\x00").rstrip(),
        }

        # Initiate reading
        self._read()

    @property
    def info(self):
        return self._info

    def _splitValues(self, x, type, splitter):
        s = x.decode("ascii").strip("\x00")
        try:
            if splitter in s:
                return tuple([type(v) for v in s.split(splitter) if v.strip()])
            else:
                return type(s)
        except ValueError:
            return s

    def _unpack(self, fmt, value):
        return struct.unpack(self._unpackPrefix + fmt, value)[0]

    # Really only so we need minimal changes to _pixel_data_numpy
    def __iter__(self):
        return iter(self._info.keys())

    def __getattr__(self, key):
        info = object.__getattribute__(self, "_info")
        if key in info:
            return info[key]
        return object.__getattribute__(self, key)  # pragma: no cover

    def _read(self):
        f = self._file
        # Check prefix after peamble
        f.seek(128)
        if f.read(4) != b"DICM":
            raise NotADicomFile("Not a valid DICOM file.")
        # Read
        self._read_header()
        self._read_data_elements()
        self._get_shape_and_sampling()
        # Close if done, reopen if necessary to read pixel data
        if os.path.isfile(self._filename):
            self._file.close()
            self._file = None

    def _readDataElement(self):
        f = self._file
        # Get group  and element
        group = self._unpack("H", f.read(2))
        element = self._unpack("H", f.read(2))
        # Get value length
        if self.is_implicit_VR:
            vl = self._unpack("I", f.read(4))
        else:
            vr = f.read(2)
            if vr in (b"OB", b"OW", b"SQ", b"UN"):
                reserved = f.read(2)  # noqa
                vl = self._unpack("I", f.read(4))
            else:
                vl = self._unpack("H", f.read(2))
        # Get value
        if group == 0x7FE0 and element == 0x0010:
            here = f.tell()
            self._pixel_data_loc = here, vl
            f.seek(here + vl)
            return group, element, b"Deferred loading of pixel data"
        else:
            if vl == 0xFFFFFFFF:
                value = self._read_undefined_length_value()
            else:
                value = f.read(vl)
            return group, element, value

    def _read_undefined_length_value(self, read_size=128):
        """Copied (in compacted form) from PyDicom
        Copyright Darcy Mason.
        """
        fp = self._file
        # data_start = fp.tell()
        search_rewind = 3
        bytes_to_find = struct.pack(
            self._unpackPrefix + "HH", SequenceDelimiterTag[0], SequenceDelimiterTag[1]
        )

        found = False
        value_chunks = []
        while not found:
            chunk_start = fp.tell()
            bytes_read = fp.read(read_size)
            if len(bytes_read) < read_size:
                # try again,
                # if still don't get required amount, this is last block
                new_bytes = fp.read(read_size - len(bytes_read))
                bytes_read += new_bytes
                if len(bytes_read) < read_size:
                    raise EOFError(
                        "End of file reached before sequence " "delimiter found."
                    )
            index = bytes_read.find(bytes_to_find)
            if index != -1:
                found = True
                value_chunks.append(bytes_read[:index])
                fp.seek(chunk_start + index + 4)  # rewind to end of delimiter
                length = fp.read(4)
                if length != b"\0\0\0\0":
                    logger.warning(
                        "Expected 4 zero bytes after undefined length " "delimiter"
                    )
            else:
                fp.seek(fp.tell() - search_rewind)  # rewind a bit
                # accumulate the bytes read (not including the rewind)
                value_chunks.append(bytes_read[:-search_rewind])

        # if get here then have found the byte string
        return b"".join(value_chunks)

    def _read_header(self):
        f = self._file
        TransferSyntaxUID = None

        # Read all elements, store transferSyntax when we encounter it
        try:
            while True:
                fp_save = f.tell()
                # Get element
                group, element, value = self._readDataElement()
                if group == 0x02:
                    if group == 0x02 and element == 0x10:
                        TransferSyntaxUID = value.decode("ascii").strip("\x00")
                else:
                    # No more group 2: rewind and break
                    # (don't trust group length)
                    f.seek(fp_save)
                    break
        except (EOFError, struct.error):  # pragma: no cover
            raise RuntimeError("End of file reached while still in header.")

        # Handle transfer syntax
        self._info["TransferSyntaxUID"] = TransferSyntaxUID
        #
        if TransferSyntaxUID is None:
            # Assume ExplicitVRLittleEndian
            is_implicit_VR, is_little_endian = False, True
        elif TransferSyntaxUID == "1.2.840.10008.1.2.1":
            # ExplicitVRLittleEndian
            is_implicit_VR, is_little_endian = False, True
        elif TransferSyntaxUID == "1.2.840.10008.1.2.2":
            # ExplicitVRBigEndian
            is_implicit_VR, is_little_endian = False, False
        elif TransferSyntaxUID == "1.2.840.10008.1.2":
            # implicit VR little endian
            is_implicit_VR, is_little_endian = True, True
        elif TransferSyntaxUID == "1.2.840.10008.1.2.1.99":
            # DeflatedExplicitVRLittleEndian:
            is_implicit_VR, is_little_endian = False, True
            self._inflate()
        else:
            # http://www.dicomlibrary.com/dicom/transfer-syntax/
            t, extra_info = TransferSyntaxUID, ""
            if "1.2.840.10008.1.2.4.50" <= t < "1.2.840.10008.1.2.4.99":
                extra_info = " (JPEG)"
            if "1.2.840.10008.1.2.4.90" <= t < "1.2.840.10008.1.2.4.99":
                extra_info = " (JPEG 2000)"
            if t == "1.2.840.10008.1.2.5":
                extra_info = " (RLE)"
            if t == "1.2.840.10008.1.2.6.1":
                extra_info = " (RFC 2557)"
            raise CompressedDicom(
                "The dicom reader can only read files with "
                "uncompressed image data - not %r%s. You "
                "can try using dcmtk or gdcm to convert the "
                "image." % (t, extra_info)
            )

        # From hereon, use implicit/explicit big/little endian
        self.is_implicit_VR = is_implicit_VR
        self.is_little_endian = is_little_endian
        self._unpackPrefix = "><"[is_little_endian]

    def _read_data_elements(self):
        info = self._info
        try:
            while True:
                # Get element
                group, element, value = self._readDataElement()
                # Is it a group we are interested in?
                if group in GROUPS:
                    key = (group, element)
                    name, vr = MINIDICT.get(key, (None, None))
                    # Is it an element we are interested in?
                    if name:
                        # Store value
                        converter = self._converters.get(vr, lambda x: x)
                        info[name] = converter(value)
        except (EOFError, struct.error):
            pass  # end of file ...

    def get_numpy_array(self):
        """Get numpy arra for this DICOM file, with the correct shape,
        and pixel values scaled appropriately.
        """
        # Is there pixel data at all?
        if "PixelData" not in self:
            raise TypeError("No pixel data found in this dataset.")

        # Load it now if it was not already loaded
        if self._pixel_data_loc and len(self.PixelData) < 100:
            # Reopen file?
            close_file = False
            if self._file is None:
                close_file = True
                self._file = open(self._filename, "rb")
            # Read data
            self._file.seek(self._pixel_data_loc[0])
            if self._pixel_data_loc[1] == 0xFFFFFFFF:
                value = self._read_undefined_length_value()
            else:
                value = self._file.read(self._pixel_data_loc[1])
            # Close file
            if close_file:
                self._file.close()
                self._file = None
            # Overwrite
            self._info["PixelData"] = value

        # Get data
        data = self._pixel_data_numpy()
        data = self._apply_slope_and_offset(data)

        # Remove data again to preserve memory
        # Note that the data for the original file is loaded twice ...
        self._info["PixelData"] = (
            b"Data converted to numpy array, " + b"raw data removed to preserve memory"
        )
        return data

    def _get_shape_and_sampling(self):
        """Get shape and sampling without actuall using the pixel data.
        In this way, the user can get an idea what's inside without having
        to load it.
        """
        # Get shape (in the same way that pydicom does)
        if "NumberOfFrames" in self and self.NumberOfFrames > 1:
            if self.SamplesPerPixel > 1:
                shape = (
                    self.SamplesPerPixel,
                    self.NumberOfFrames,
                    self.Rows,
                    self.Columns,
                )
            else:
                shape = self.NumberOfFrames, self.Rows, self.Columns
        elif "SamplesPerPixel" in self:
            if self.SamplesPerPixel > 1:
                if self.BitsAllocated == 8:
                    shape = self.SamplesPerPixel, self.Rows, self.Columns
                else:
                    raise NotImplementedError(
                        "DICOM plugin only handles "
                        "SamplesPerPixel > 1 if Bits "
                        "Allocated = 8"
                    )
            else:
                shape = self.Rows, self.Columns
        else:
            raise RuntimeError(
                "DICOM file has no SamplesPerPixel " "(perhaps this is a report?)"
            )

        # Try getting sampling between pixels
        if "PixelSpacing" in self:
            sampling = float(self.PixelSpacing[0]), float(self.PixelSpacing[1])
        else:
            sampling = 1.0, 1.0
        if "SliceSpacing" in self:
            sampling = (abs(self.SliceSpacing),) + sampling

        # Ensure that sampling has as many elements as shape
        sampling = (1.0,) * (len(shape) - len(sampling)) + sampling[-len(shape) :]

        # Set shape and sampling
        self._info["shape"] = shape
        self._info["sampling"] = sampling

    def _pixel_data_numpy(self):
        """Return a NumPy array of the pixel data."""
        # Taken from pydicom
        # Copyright (c) 2008-2012 Darcy Mason

        if "PixelData" not in self:
            raise TypeError("No pixel data found in this dataset.")

        # determine the type used for the array
        need_byteswap = self.is_little_endian != sys_is_little_endian

        # Make NumPy format code, e.g. "uint16", "int32" etc
        # from two pieces of info:
        #    self.PixelRepresentation -- 0 for unsigned, 1 for signed;
        #    self.BitsAllocated -- 8, 16, or 32
        format_str = "%sint%d" % (
            ("u", "")[self.PixelRepresentation],
            self.BitsAllocated,
        )
        try:
            numpy_format = np.dtype(format_str)
        except TypeError:  # pragma: no cover
            raise TypeError(
                "Data type not understood by NumPy: format='%s', "
                " PixelRepresentation=%d, BitsAllocated=%d"
                % (numpy_format, self.PixelRepresentation, self.BitsAllocated)
            )

        # Have correct Numpy format, so create the NumPy array
        arr = np.frombuffer(self.PixelData, numpy_format).copy()

        # XXX byte swap - may later handle this in read_file!!?
        if need_byteswap:
            arr.byteswap(True)  # True means swap in-place, don't make new copy

        # Note the following reshape operations return a new *view* onto arr,
        # but don't copy the data
        arr = arr.reshape(*self._info["shape"])
        return arr

    def _apply_slope_and_offset(self, data):
        """
        If RescaleSlope and RescaleIntercept are present in the data,
        apply them. The data type of the data is changed if necessary.
        """
        # Obtain slope and offset
        slope, offset = 1, 0
        needFloats, needApplySlopeOffset = False, False
        if "RescaleSlope" in self:
            needApplySlopeOffset = True
            slope = self.RescaleSlope
        if "RescaleIntercept" in self:
            needApplySlopeOffset = True
            offset = self.RescaleIntercept
        if int(slope) != slope or int(offset) != offset:
            needFloats = True
        if not needFloats:
            slope, offset = int(slope), int(offset)

        # Apply slope and offset
        if needApplySlopeOffset:
            # Maybe we need to change the datatype?
            if data.dtype in [np.float32, np.float64]:
                pass
            elif needFloats:
                data = data.astype(np.float32)
            else:
                # Determine required range
                minReq, maxReq = data.min(), data.max()
                minReq = min([minReq, minReq * slope + offset, maxReq * slope + offset])
                maxReq = max([maxReq, minReq * slope + offset, maxReq * slope + offset])

                # Determine required datatype from that
                dtype = None
                if minReq < 0:
                    # Signed integer type
                    maxReq = max([-minReq, maxReq])
                    if maxReq < 2 ** 7:
                        dtype = np.int8
                    elif maxReq < 2 ** 15:
                        dtype = np.int16
                    elif maxReq < 2 ** 31:
                        dtype = np.int32
                    else:
                        dtype = np.float32
                else:
                    # Unsigned integer type
                    if maxReq < 2 ** 8:
                        dtype = np.int8
                    elif maxReq < 2 ** 16:
                        dtype = np.int16
                    elif maxReq < 2 ** 32:
                        dtype = np.int32
                    else:
                        dtype = np.float32
                # Change datatype
                if dtype != data.dtype:
                    data = data.astype(dtype)

            # Apply slope and offset
            data *= slope
            data += offset

        # Done
        return data

    def _inflate(self):
        # Taken from pydicom
        # Copyright (c) 2008-2012 Darcy Mason
        import zlib
        from io import BytesIO

        # See PS3.6-2008 A.5 (p 71) -- when written, the entire dataset
        #   following the file metadata was prepared the normal way,
        #   then "deflate" compression applied.
        #  All that is needed here is to decompress and then
        #      use as normal in a file-like object
        zipped = self._file.read()
        # -MAX_WBITS part is from comp.lang.python answer:
        # groups.google.com/group/comp.lang.python/msg/e95b3b38a71e6799
        unzipped = zlib.decompress(zipped, -zlib.MAX_WBITS)
        self._file = BytesIO(unzipped)  # a file-like object


class DicomSeries(object):
    """DicomSeries
    This class represents a serie of dicom files (SimpleDicomReader
    objects) that belong together. If these are multiple files, they
    represent the slices of a volume (like for CT or MRI).
    """

    def __init__(self, suid, progressIndicator):
        # Init dataset list and the callback
        self._entries = []

        # Init props
        self._suid = suid
        self._info = {}
        self._progressIndicator = progressIndicator

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index):
        return self._entries[index]

    @property
    def suid(self):
        return self._suid

    @property
    def shape(self):
        """The shape of the data (nz, ny, nx)."""
        return self._info["shape"]

    @property
    def sampling(self):
        """The sampling (voxel distances) of the data (dz, dy, dx)."""
        return self._info["sampling"]

    @property
    def info(self):
        """A dictionary containing the information as present in the
        first dicomfile of this serie. None if there are no entries."""
        return self._info

    @property
    def description(self):
        """A description of the dicom series. Used fields are
        PatientName, shape of the data, SeriesDescription, and
        ImageComments.
        """
        info = self.info

        # If no info available, return simple description
        if not info:  # pragma: no cover
            return "DicomSeries containing %i images" % len(self)

        fields = []
        # Give patient name
        if "PatientName" in info:
            fields.append("" + info["PatientName"])
        # Also add dimensions
        if self.shape:
            tmp = [str(d) for d in self.shape]
            fields.append("x".join(tmp))
        # Try adding more fields
        if "SeriesDescription" in info:
            fields.append("'" + info["SeriesDescription"] + "'")
        if "ImageComments" in info:
            fields.append("'" + info["ImageComments"] + "'")

        # Combine
        return " ".join(fields)

    def __repr__(self):
        adr = hex(id(self)).upper()
        return "<DicomSeries with %i images at %s>" % (len(self), adr)

    def get_numpy_array(self):
        """Get (load) the data that this DicomSeries represents, and return
        it as a numpy array. If this serie contains multiple images, the
        resulting array is 3D, otherwise it's 2D.
        """

        # It's easy if no file or if just a single file
        if len(self) == 0:
            raise ValueError("Serie does not contain any files.")
        elif len(self) == 1:
            return self[0].get_numpy_array()

        # Check info
        if self.info is None:
            raise RuntimeError("Cannot return volume if series not finished.")

        # Init data (using what the dicom packaged produces as a reference)
        slice = self[0].get_numpy_array()
        vol = np.zeros(self.shape, dtype=slice.dtype)
        vol[0] = slice

        # Fill volume
        self._progressIndicator.start("loading data", "", len(self))
        for z in range(1, len(self)):
            vol[z] = self[z].get_numpy_array()
            self._progressIndicator.set_progress(z + 1)
        self._progressIndicator.finish()

        # Done
        import gc

        gc.collect()
        return vol

    def _append(self, dcm):
        self._entries.append(dcm)

    def _sort(self):
        self._entries.sort(key=lambda k: k.InstanceNumber)

    def _finish(self):
        """
        Evaluate the series of dicom files. Together they should make up
        a volumetric dataset. This means the files should meet certain
        conditions. Also some additional information has to be calculated,
        such as the distance between the slices. This method sets the
        attributes for "shape", "sampling" and "info".

        This method checks:
          * that there are no missing files
          * that the dimensions of all images match
          * that the pixel spacing of all images match
        """

        # The datasets list should be sorted by instance number
        L = self._entries
        if len(L) == 0:
            return
        elif len(L) == 1:
            self._info = L[0].info
            return

        # Get previous
        ds1 = L[0]
        # Init measures to calculate average of
        distance_sum = 0.0
        # Init measures to check (these are in 2D)
        dimensions = ds1.Rows, ds1.Columns
        # sampling = float(ds1.PixelSpacing[0]), float(ds1.PixelSpacing[1])
        sampling = ds1.info["sampling"][:2]  # row, column

        for index in range(len(L)):
            # The first round ds1 and ds2 will be the same, for the
            # distance calculation this does not matter
            # Get current
            ds2 = L[index]
            # Get positions
            pos1 = float(ds1.ImagePositionPatient[2])
            pos2 = float(ds2.ImagePositionPatient[2])
            # Update distance_sum to calculate distance later
            distance_sum += abs(pos1 - pos2)
            # Test measures
            dimensions2 = ds2.Rows, ds2.Columns
            # sampling2 = float(ds2.PixelSpacing[0]), float(ds2.PixelSpacing[1])
            sampling2 = ds2.info["sampling"][:2]  # row, column
            if dimensions != dimensions2:
                # We cannot produce a volume if the dimensions match
                raise ValueError("Dimensions of slices does not match.")
            if sampling != sampling2:
                # We can still produce a volume, but we should notify the user
                self._progressIndicator.write("Warn: sampling does not match.")
            # Store previous
            ds1 = ds2

        # Finish calculating average distance
        # (Note that there are len(L)-1 distances)
        distance_mean = distance_sum / (len(L) - 1)

        # Set info dict
        self._info = L[0].info.copy()

        # Store information that is specific for the serie
        self._info["shape"] = (len(L),) + ds2.info["shape"]
        self._info["sampling"] = (distance_mean,) + ds2.info["sampling"]


def list_files(files, path):
    """List all files in the directory, recursively."""
    for item in os.listdir(path):
        item = os.path.join(path, item)
        if os.path.isdir(item):
            list_files(files, item)
        elif os.path.isfile(item):
            files.append(item)


def process_directory(request, progressIndicator, readPixelData=False):
    """
    Reads dicom files and returns a list of DicomSeries objects, which
    contain information about the data, and can be used to load the
    image or volume data.

    if readPixelData is True, the pixel data of all series is read. By
    default the loading of pixeldata is deferred until it is requested
    using the DicomSeries.get_pixel_array() method. In general, both
    methods should be equally fast.
    """
    # Get directory to examine
    if os.path.isdir(request.filename):
        path = request.filename
    elif os.path.isfile(request.filename):
        path = os.path.dirname(request.filename)
    else:  # pragma: no cover - tested earlier
        raise ValueError(
            "Dicom plugin needs a valid filename to examine " "the directory"
        )

    # Check files
    files = []
    list_files(files, path)  # Find files recursively

    # Gather file data and put in DicomSeries
    series = {}
    count = 0
    progressIndicator.start("examining files", "files", len(files))
    for filename in files:
        # Show progress (note that we always start with a 0.0)
        count += 1
        progressIndicator.set_progress(count)
        # Skip DICOMDIR files
        if filename.count("DICOMDIR"):  # pragma: no cover
            continue
        # Try loading dicom ...
        try:
            dcm = SimpleDicomReader(filename)
        except NotADicomFile:
            continue  # skip non-dicom file
        except Exception as why:  # pragma: no cover
            progressIndicator.write(str(why))
            continue
        # Get SUID and register the file with an existing or new series object
        try:
            suid = dcm.SeriesInstanceUID
        except AttributeError:  # pragma: no cover
            continue  # some other kind of dicom file
        if suid not in series:
            series[suid] = DicomSeries(suid, progressIndicator)
        series[suid]._append(dcm)

    # Finish progress
    # progressIndicator.finish('Found %i series.' % len(series))

    # Make a list and sort, so that the order is deterministic
    series = list(series.values())
    series.sort(key=lambda x: x.suid)

    # Split series if necessary
    for serie in reversed([serie for serie in series]):
        splitSerieIfRequired(serie, series, progressIndicator)

    # Finish all series
    # progressIndicator.start('analyse series', '', len(series))
    series_ = []
    for i in range(len(series)):
        try:
            series[i]._finish()
            series_.append(series[i])
        except Exception as err:  # pragma: no cover
            progressIndicator.write(str(err))
            pass  # Skip serie (probably report-like file without pixels)
        # progressIndicator.set_progress(i+1)
    progressIndicator.finish("Found %i correct series." % len(series_))

    # Done
    return series_


def splitSerieIfRequired(serie, series, progressIndicator):
    """
    Split the serie in multiple series if this is required. The choice
    is based on examing the image position relative to the previous
    image. If it differs too much, it is assumed that there is a new
    dataset. This can happen for example in unspitted gated CT data.
    """

    # Sort the original list and get local name
    serie._sort()
    L = serie._entries
    # Init previous slice
    ds1 = L[0]
    # Check whether we can do this
    if "ImagePositionPatient" not in ds1:
        return
    # Initialize a list of new lists
    L2 = [[ds1]]
    # Init slice distance estimate
    distance = 0

    for index in range(1, len(L)):
        # Get current slice
        ds2 = L[index]
        # Get positions
        pos1 = float(ds1.ImagePositionPatient[2])
        pos2 = float(ds2.ImagePositionPatient[2])
        # Get distances
        newDist = abs(pos1 - pos2)
        # deltaDist = abs(firstPos-pos2)
        # If the distance deviates more than 2x from what we've seen,
        # we can agree it's a new dataset.
        if distance and newDist > 2.1 * distance:
            L2.append([])
            distance = 0
        else:
            # Test missing file
            if distance and newDist > 1.5 * distance:
                progressIndicator.write(
                    "Warning: missing file after %r" % ds1._filename
                )
            distance = newDist
        # Add to last list
        L2[-1].append(ds2)
        # Store previous
        ds1 = ds2

    # Split if we should
    if len(L2) > 1:
        # At what position are we now?
        i = series.index(serie)
        # Create new series
        series2insert = []
        for L in L2:
            newSerie = DicomSeries(serie.suid, progressIndicator)
            newSerie._entries = L
            series2insert.append(newSerie)
        # Insert series and remove self
        for newSerie in reversed(series2insert):
            series.insert(i, newSerie)
        series.remove(serie)
