# -*- coding: utf-8 -*-
# Copyright (c) 2018, imageio contributors
# imageio is distributed under the terms of the (new) BSD License.
#

""" Read LFR files (Lytro Illum).

Backend: internal

Plugin to read Lytro Illum .lfr and .raw files as produced
by the Lytro Illum light field camera. It is actually a collection
of plugins, each supporting slightly different keyword arguments

Parameters
----------
meta_only : bool
    Whether to only read the metadata.
include_thumbnail : bool
    (only for lytro-lfr and lytro-lfp)
    Whether to include an image thumbnail in the metadata.

"""
#
#
# This code is based on work by
# David Uhlig and his lfr_reader
#   (https://www.iiit.kit.edu/uhlig.php)
# Donald Dansereau and his Matlab LF Toolbox
#   (http://dgd.vision/Tools/LFToolbox/)
# and Behnam Esfahbod and his Python LFP-Reader
#   (https://github.com/behnam/python-lfp-reader/)


import os
import json
import struct
import logging


import numpy as np

from ..core import Format
from .. import imread


logger = logging.getLogger(__name__)


# Sensor size of Lytro Illum resp. Lytro F01 light field camera sensor
LYTRO_ILLUM_IMAGE_SIZE = (5368, 7728)
LYTRO_F01_IMAGE_SIZE = (3280, 3280)

# Parameter of lfr file format
HEADER_LENGTH = 12
SIZE_LENGTH = 4  # = 16 - header_length
SHA1_LENGTH = 45  # = len("sha1-") + (160 / 4)
PADDING_LENGTH = 35  # = (4*16) - header_length - size_length - sha1_length
DATA_CHUNKS_ILLUM = 11
DATA_CHUNKS_F01 = 3


class LytroFormat(Format):
    """Base class for Lytro format.
    The subclasses LytroLfrFormat, LytroLfpFormat, LytroIllumRawFormat and
    LytroF01RawFormat implement the Lytro-LFR, Lytro-LFP and Lytro-RAW format
    for the Illum and original F01 camera respectively.
    Writing is not supported.
    """

    # Only single images are supported.
    _modes = "i"

    def _can_write(self, request):
        # Writing of Lytro files is not supported
        return False

    # -- writer

    class Writer(Format.Writer):
        def _open(self, flags=0):
            self._fp = self.request.get_file()

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._fp
            pass

        def _append_data(self, im, meta):
            # Process the given data and meta data.
            raise RuntimeError("The lytro format cannot write image data.")

        def _set_meta_data(self, meta):
            # Process the given meta data (global for all images)
            # It is not mandatory to support this.
            raise RuntimeError("The lytro format cannot write meta data.")


class LytroIllumRawFormat(LytroFormat):
    """This is the Lytro Illum RAW format.
    The raw format is a 10bit image format as used by the Lytro Illum
    light field camera. The format will read the specified raw file and will
    try to load a .txt or .json file with the associated meta data.
    This format does not support writing.


    Parameters for reading
    ----------------------
    meta_only : bool
        Whether to only read the metadata.
    """

    def _can_read(self, request):
        # Check if mode and extensions are supported by the format
        if request.mode[1] in (self.modes + "?"):
            if request.extension in (".raw",):
                return True

    @staticmethod
    def rearrange_bits(array):
        # Do bit rearrangement for the 10-bit lytro raw format
        # Normalize output to 1.0 as float64
        t0 = array[0::5]
        t1 = array[1::5]
        t2 = array[2::5]
        t3 = array[3::5]
        lsb = array[4::5]

        t0 = np.left_shift(t0, 2) + np.bitwise_and(lsb, 3)
        t1 = np.left_shift(t1, 2) + np.right_shift(np.bitwise_and(lsb, 12), 2)
        t2 = np.left_shift(t2, 2) + np.right_shift(np.bitwise_and(lsb, 48), 4)
        t3 = np.left_shift(t3, 2) + np.right_shift(np.bitwise_and(lsb, 192), 6)

        image = np.zeros(LYTRO_ILLUM_IMAGE_SIZE, dtype=np.uint16)
        image[:, 0::4] = t0.reshape(
            (LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4)
        )
        image[:, 1::4] = t1.reshape(
            (LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4)
        )
        image[:, 2::4] = t2.reshape(
            (LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4)
        )
        image[:, 3::4] = t3.reshape(
            (LYTRO_ILLUM_IMAGE_SIZE[0], LYTRO_ILLUM_IMAGE_SIZE[1] // 4)
        )

        # Normalize data to 1.0 as 64-bit float.
        # Division is by 1023 as the Lytro Illum saves 10-bit raw data.
        return np.divide(image, 1023.0).astype(np.float64)

    # -- reader

    class Reader(Format.Reader):
        def _open(self, meta_only=False):
            self._file = self.request.get_file()
            self._data = None
            self._meta_only = meta_only

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._file
            del self._data

        def _get_length(self):
            # Return the number of images.
            return 1

        def _get_data(self, index):
            # Return the data and meta data for the given index

            if index not in [0, "None"]:
                raise IndexError("Lytro file contains only one dataset")

            if not self._meta_only:
                # Read all bytes
                if self._data is None:
                    self._data = self._file.read()

                # Read bytes from string and convert to uint16
                raw = np.frombuffer(self._data, dtype=np.uint8).astype(np.uint16)

                # Rearrange bits
                img = LytroIllumRawFormat.rearrange_bits(raw)

            else:
                # Return empty image
                img = np.array([])

            # Return image and meta data
            return img, self._get_meta_data(index=0)

        def _get_meta_data(self, index):
            # Get the meta data for the given index. If index is None, it
            # should return the global meta data.

            if index not in [0, None]:
                raise IndexError("Lytro meta data file contains only one dataset")

            # Try to read meta data from meta data file corresponding
            # to the raw data file, extension in [.txt, .TXT, .json, .JSON]
            filename_base = os.path.splitext(self.request.get_local_filename())[0]

            meta_data = None

            for ext in [".txt", ".TXT", ".json", ".JSON"]:
                if os.path.isfile(filename_base + ext):
                    meta_data = json.load(open(filename_base + ext))

            if meta_data is not None:
                return meta_data

            else:
                logger.warning("No metadata file found for provided raw file.")
                return {}


class LytroLfrFormat(LytroFormat):
    """This is the Lytro Illum LFR format.
    The lfr is a image and meta data container format as used by the
    Lytro Illum light field camera.
    The format will read the specified lfr file.
    This format does not support writing.

    Parameters for reading
    ----------------------
    meta_only : bool
        Whether to only read the metadata.
    include_thumbnail : bool
        Whether to include an image thumbnail in the metadata.
    """

    def _can_read(self, request):
        # Check if mode and extensions are supported by the format
        if request.mode[1] in (self.modes + "?"):
            if request.extension in (".lfr",):
                return True

    # -- reader

    class Reader(Format.Reader):
        def _open(self, meta_only=False, include_thumbnail=True):
            self._file = self.request.get_file()
            self._data = None
            self._chunks = {}
            self.metadata = {}
            self._content = None
            self._meta_only = meta_only
            self._include_thumbnail = include_thumbnail

            self._find_header()
            self._find_chunks()
            self._find_meta()

            try:
                # Get sha1 dict and check if it is in dictionary of data chunks
                chunk_dict = self._content["frames"][0]["frame"]
                if (
                    chunk_dict["metadataRef"] in self._chunks
                    and chunk_dict["imageRef"] in self._chunks
                    and chunk_dict["privateMetadataRef"] in self._chunks
                ):
                    if not self._meta_only:
                        # Read raw image data byte buffer
                        data_pos, size = self._chunks[chunk_dict["imageRef"]]
                        self._file.seek(data_pos, 0)
                        self.raw_image_data = self._file.read(size)

                    # Read meta data
                    data_pos, size = self._chunks[chunk_dict["metadataRef"]]
                    self._file.seek(data_pos, 0)
                    metadata = self._file.read(size)
                    # Add metadata to meta data dict
                    self.metadata["metadata"] = json.loads(metadata.decode("ASCII"))

                    # Read private metadata
                    data_pos, size = self._chunks[chunk_dict["privateMetadataRef"]]
                    self._file.seek(data_pos, 0)
                    serial_numbers = self._file.read(size)
                    self.serial_numbers = json.loads(serial_numbers.decode("ASCII"))
                    # Add private metadata to meta data dict
                    self.metadata["privateMetadata"] = self.serial_numbers

                # Read image preview thumbnail
                if self._include_thumbnail:
                    chunk_dict = self._content["thumbnails"][0]
                    if chunk_dict["imageRef"] in self._chunks:
                        # Read thumbnail image from thumbnail chunk
                        data_pos, size = self._chunks[chunk_dict["imageRef"]]
                        self._file.seek(data_pos, 0)
                        # Read binary data, read image as jpeg
                        thumbnail_data = self._file.read(size)
                        thumbnail_img = imread(thumbnail_data, format="jpeg")

                        thumbnail_height = chunk_dict["height"]
                        thumbnail_width = chunk_dict["width"]

                        # Add thumbnail to metadata
                        self.metadata["thumbnail"] = {
                            "image": thumbnail_img,
                            "height": thumbnail_height,
                            "width": thumbnail_width,
                        }

            except KeyError:
                raise RuntimeError("The specified file is not a valid LFR file.")

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._file
            del self._data

        def _get_length(self):
            # Return the number of images. Can be np.inf
            return 1

        def _find_header(self):
            """
            Checks if file has correct header and skip it.
            """
            file_header = b"\x89LFP\x0D\x0A\x1A\x0A\x00\x00\x00\x01"
            # Read and check header of file
            header = self._file.read(HEADER_LENGTH)
            if header != file_header:
                raise RuntimeError("The LFR file header is invalid.")

            # Read first bytes to skip header
            self._file.read(SIZE_LENGTH)

        def _find_chunks(self):
            """
            Gets start position and size of data chunks in file.
            """
            chunk_header = b"\x89LFC\x0D\x0A\x1A\x0A\x00\x00\x00\x00"

            for i in range(0, DATA_CHUNKS_ILLUM):
                data_pos, size, sha1 = self._get_chunk(chunk_header)
                self._chunks[sha1] = (data_pos, size)

        def _find_meta(self):
            """
            Gets a data chunk that contains information over content
            of other data chunks.
            """
            meta_header = b"\x89LFM\x0D\x0A\x1A\x0A\x00\x00\x00\x00"
            data_pos, size, sha1 = self._get_chunk(meta_header)

            # Get content
            self._file.seek(data_pos, 0)
            data = self._file.read(size)
            self._content = json.loads(data.decode("ASCII"))

        def _get_chunk(self, header):
            """
            Checks if chunk has correct header and skips it.
            Finds start position and length of next chunk and reads
            sha1-string that identifies the following data chunk.

            Parameters
            ----------
            header : bytes
                Byte string that identifies start of chunk.

            Returns
            -------
                data_pos : int
                    Start position of data chunk in file.
                size : int
                    Size of data chunk.
                sha1 : str
                    Sha1 value of chunk.
            """
            # Read and check header of chunk
            header_chunk = self._file.read(HEADER_LENGTH)
            if header_chunk != header:
                raise RuntimeError("The LFR chunk header is invalid.")

            data_pos = None
            sha1 = None

            # Read size
            size = struct.unpack(">i", self._file.read(SIZE_LENGTH))[0]
            if size > 0:
                # Read sha1
                sha1 = str(self._file.read(SHA1_LENGTH).decode("ASCII"))
                # Skip fixed null chars
                self._file.read(PADDING_LENGTH)
                # Find start of data and skip data
                data_pos = self._file.tell()
                self._file.seek(size, 1)
                # Skip extra null chars
                ch = self._file.read(1)
                while ch == b"\0":
                    ch = self._file.read(1)
                self._file.seek(-1, 1)

            return data_pos, size, sha1

        def _get_data(self, index):
            # Return the data and meta data for the given index
            if index not in [0, None]:
                raise IndexError("Lytro lfr file contains only one dataset")

            if not self._meta_only:
                # Read bytes from string and convert to uint16
                raw = np.frombuffer(self.raw_image_data, dtype=np.uint8).astype(
                    np.uint16
                )
                im = LytroIllumRawFormat.rearrange_bits(raw)
            else:
                im = np.array([])

            # Return array and dummy meta data
            return im, self.metadata

        def _get_meta_data(self, index):
            # Get the meta data for the given index. If index is None,
            # it returns the global meta data.
            if index not in [0, None]:
                raise IndexError("Lytro meta data file contains only one dataset")

            return self.metadata


class LytroF01RawFormat(LytroFormat):
    """This is the Lytro RAW format for the original F01 Lytro camera.
    The raw format is a 12bit image format as used by the Lytro F01
    light field camera. The format will read the specified raw file and will
    try to load a .txt or .json file with the associated meta data.
    This format does not support writing.


    Parameters for reading
    ----------------------
    meta_only : bool
        Whether to only read the metadata.

    """

    def _can_read(self, request):
        # Check if mode and extensions are supported by the format
        if request.mode[1] in (self.modes + "?"):
            if request.extension in (".raw",):
                return True

    @staticmethod
    def rearrange_bits(array):
        # Do bit rearrangement for the 12-bit lytro raw format
        # Normalize output to 1.0 as float64
        t0 = array[0::3]
        t1 = array[1::3]
        t2 = array[2::3]

        a0 = np.left_shift(t0, 4) + np.right_shift(np.bitwise_and(t1, 240), 4)
        a1 = np.left_shift(np.bitwise_and(t1, 15), 8) + t2

        image = np.zeros(LYTRO_F01_IMAGE_SIZE, dtype=np.uint16)
        image[:, 0::2] = a0.reshape(
            (LYTRO_F01_IMAGE_SIZE[0], LYTRO_F01_IMAGE_SIZE[1] // 2)
        )
        image[:, 1::2] = a1.reshape(
            (LYTRO_F01_IMAGE_SIZE[0], LYTRO_F01_IMAGE_SIZE[1] // 2)
        )

        # Normalize data to 1.0 as 64-bit float.
        # Division is by 4095 as the Lytro F01 saves 12-bit raw data.
        return np.divide(image, 4095.0).astype(np.float64)

    # -- reader

    class Reader(Format.Reader):
        def _open(self, meta_only=False):
            self._file = self.request.get_file()
            self._data = None
            self._meta_only = meta_only

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._file
            del self._data

        def _get_length(self):
            # Return the number of images.
            return 1

        def _get_data(self, index):
            # Return the data and meta data for the given index

            if index not in [0, "None"]:
                raise IndexError("Lytro file contains only one dataset")

            if not self._meta_only:
                # Read all bytes
                if self._data is None:
                    self._data = self._file.read()

                # Read bytes from string and convert to uint16
                raw = np.frombuffer(self._data, dtype=np.uint8).astype(np.uint16)

                # Rearrange bits
                img = LytroF01RawFormat.rearrange_bits(raw)

            else:
                img = np.array([])

            # Return image and meta data
            return img, self._get_meta_data(index=0)

        def _get_meta_data(self, index):
            # Get the meta data for the given index. If index is None, it
            # should return the global meta data.

            if index not in [0, None]:
                raise IndexError("Lytro meta data file contains only one dataset")

            # Try to read meta data from meta data file corresponding
            # to the raw data file, extension in [.txt, .TXT, .json, .JSON]
            filename_base = os.path.splitext(self.request.get_local_filename())[0]

            meta_data = None

            for ext in [".txt", ".TXT", ".json", ".JSON"]:
                if os.path.isfile(filename_base + ext):
                    meta_data = json.load(open(filename_base + ext))

            if meta_data is not None:
                return meta_data

            else:
                logger.warning("No metadata file found for provided raw file.")
                return {}


class LytroLfpFormat(LytroFormat):
    """This is the Lytro Illum LFP format.
    The lfp is a image and meta data container format as used by the
    Lytro F01 light field camera.
    The format will read the specified lfp file.
    This format does not support writing.

    Parameters for reading
    ----------------------
    meta_only : bool
        Whether to only read the metadata.
    include_thumbnail : bool
        Whether to include an image thumbnail in the metadata.
    """

    def _can_read(self, request):
        # Check if mode and extensions are supported by the format
        if request.mode[1] in (self.modes + "?"):
            if request.extension in (".lfp",):
                return True

    # -- reader

    class Reader(Format.Reader):
        def _open(self, meta_only=False):
            self._file = self.request.get_file()
            self._data = None
            self._chunks = {}
            self.metadata = {}
            self._content = None
            self._meta_only = meta_only

            self._find_header()
            self._find_meta()
            self._find_chunks()

            try:
                # Get sha1 dict and check if it is in dictionary of data chunks
                chunk_dict = self._content["picture"]["frameArray"][0]["frame"]
                if (
                    chunk_dict["metadataRef"] in self._chunks
                    and chunk_dict["imageRef"] in self._chunks
                    and chunk_dict["privateMetadataRef"] in self._chunks
                ):
                    if not self._meta_only:
                        # Read raw image data byte buffer
                        data_pos, size = self._chunks[chunk_dict["imageRef"]]
                        self._file.seek(data_pos, 0)
                        self.raw_image_data = self._file.read(size)

                    # Read meta data
                    data_pos, size = self._chunks[chunk_dict["metadataRef"]]
                    self._file.seek(data_pos, 0)
                    metadata = self._file.read(size)
                    # Add metadata to meta data dict
                    self.metadata["metadata"] = json.loads(metadata.decode("ASCII"))

                    # Read private metadata
                    data_pos, size = self._chunks[chunk_dict["privateMetadataRef"]]
                    self._file.seek(data_pos, 0)
                    serial_numbers = self._file.read(size)
                    self.serial_numbers = json.loads(serial_numbers.decode("ASCII"))
                    # Add private metadata to meta data dict
                    self.metadata["privateMetadata"] = self.serial_numbers

            except KeyError:
                raise RuntimeError("The specified file is not a valid LFP file.")

        def _close(self):
            # Close the reader.
            # Note that the request object will close self._file
            del self._data

        def _get_length(self):
            # Return the number of images. Can be np.inf
            return 1

        def _find_header(self):
            """
            Checks if file has correct header and skip it.
            """
            file_header = b"\x89LFP\x0D\x0A\x1A\x0A\x00\x00\x00\x01"

            # Read and check header of file
            header = self._file.read(HEADER_LENGTH)
            if header != file_header:
                raise RuntimeError("The LFP file header is invalid.")

            # Read first bytes to skip header
            self._file.read(SIZE_LENGTH)

        def _find_chunks(self):
            """
            Gets start position and size of data chunks in file.
            """
            chunk_header = b"\x89LFC\x0D\x0A\x1A\x0A\x00\x00\x00\x00"

            for i in range(0, DATA_CHUNKS_F01):
                data_pos, size, sha1 = self._get_chunk(chunk_header)
                self._chunks[sha1] = (data_pos, size)

        def _find_meta(self):
            """
            Gets a data chunk that contains information over content
            of other data chunks.
            """
            meta_header = b"\x89LFM\x0D\x0A\x1A\x0A\x00\x00\x00\x00"

            data_pos, size, sha1 = self._get_chunk(meta_header)

            # Get content
            self._file.seek(data_pos, 0)
            data = self._file.read(size)
            self._content = json.loads(data.decode("ASCII"))
            data = self._file.read(5)  # Skip 5

        def _get_chunk(self, header):
            """
            Checks if chunk has correct header and skips it.
            Finds start position and length of next chunk and reads
            sha1-string that identifies the following data chunk.

            Parameters
            ----------
            header : bytes
                Byte string that identifies start of chunk.

            Returns
            -------
                data_pos : int
                    Start position of data chunk in file.
                size : int
                    Size of data chunk.
                sha1 : str
                    Sha1 value of chunk.
            """
            # Read and check header of chunk
            header_chunk = self._file.read(HEADER_LENGTH)
            if header_chunk != header:
                raise RuntimeError("The LFP chunk header is invalid.")

            data_pos = None
            sha1 = None

            # Read size
            size = struct.unpack(">i", self._file.read(SIZE_LENGTH))[0]
            if size > 0:
                # Read sha1
                sha1 = str(self._file.read(SHA1_LENGTH).decode("ASCII"))
                # Skip fixed null chars
                self._file.read(PADDING_LENGTH)
                # Find start of data and skip data
                data_pos = self._file.tell()
                self._file.seek(size, 1)
                # Skip extra null chars
                ch = self._file.read(1)
                while ch == b"\0":
                    ch = self._file.read(1)
                self._file.seek(-1, 1)

            return data_pos, size, sha1

        def _get_data(self, index):
            # Return the data and meta data for the given index
            if index not in [0, None]:
                raise IndexError("Lytro lfp file contains only one dataset")

            if not self._meta_only:
                # Read bytes from string and convert to uint16
                raw = np.frombuffer(self.raw_image_data, dtype=np.uint8).astype(
                    np.uint16
                )
                im = LytroF01RawFormat.rearrange_bits(raw)
            else:
                im = np.array([])

            # Return array and dummy meta data
            return im, self.metadata

        def _get_meta_data(self, index):
            # Get the meta data for the given index. If index is None,
            # it returns the global meta data.
            if index not in [0, None]:
                raise IndexError("Lytro meta data file contains only one dataset")

            return self.metadata
