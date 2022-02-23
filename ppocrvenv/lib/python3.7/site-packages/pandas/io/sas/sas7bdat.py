"""
Read SAS7BDAT files

Based on code written by Jared Hobbs:
  https://bitbucket.org/jaredhobbs/sas7bdat

See also:
  https://github.com/BioStatMatt/sas7bdat

Partial documentation of the file format:
  https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf

Reference for binary data compression:
  http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
"""
from __future__ import annotations

from collections import abc
from datetime import (
    datetime,
    timedelta,
)
import struct
from typing import (
    IO,
    Any,
    cast,
)

import numpy as np

from pandas.errors import (
    EmptyDataError,
    OutOfBoundsDatetime,
)

import pandas as pd
from pandas import (
    DataFrame,
    isna,
)

from pandas.io.common import get_handle
from pandas.io.sas._sas import Parser
import pandas.io.sas.sas_constants as const
from pandas.io.sas.sasreader import ReaderBase


def _parse_datetime(sas_datetime: float, unit: str):
    if isna(sas_datetime):
        return pd.NaT

    if unit == "s":
        return datetime(1960, 1, 1) + timedelta(seconds=sas_datetime)

    elif unit == "d":
        return datetime(1960, 1, 1) + timedelta(days=sas_datetime)

    else:
        raise ValueError("unit must be 'd' or 's'")


def _convert_datetimes(sas_datetimes: pd.Series, unit: str) -> pd.Series:
    """
    Convert to Timestamp if possible, otherwise to datetime.datetime.
    SAS float64 lacks precision for more than ms resolution so the fit
    to datetime.datetime is ok.

    Parameters
    ----------
    sas_datetimes : {Series, Sequence[float]}
       Dates or datetimes in SAS
    unit : {str}
       "d" if the floats represent dates, "s" for datetimes

    Returns
    -------
    Series
       Series of datetime64 dtype or datetime.datetime.
    """
    try:
        return pd.to_datetime(sas_datetimes, unit=unit, origin="1960-01-01")
    except OutOfBoundsDatetime:
        s_series = sas_datetimes.apply(_parse_datetime, unit=unit)
        s_series = cast(pd.Series, s_series)
        return s_series


class _SubheaderPointer:
    offset: int
    length: int
    compression: int
    ptype: int

    def __init__(self, offset: int, length: int, compression: int, ptype: int):
        self.offset = offset
        self.length = length
        self.compression = compression
        self.ptype = ptype


class _Column:
    col_id: int
    name: str | bytes
    label: str | bytes
    format: str | bytes  # TODO: i think allowing bytes is from py2 days
    ctype: bytes
    length: int

    def __init__(
        self,
        col_id: int,
        name: str | bytes,
        label: str | bytes,
        format: str | bytes,
        ctype: bytes,
        length: int,
    ):
        self.col_id = col_id
        self.name = name
        self.label = label
        self.format = format
        self.ctype = ctype
        self.length = length


# SAS7BDAT represents a SAS data file in SAS7BDAT format.
class SAS7BDATReader(ReaderBase, abc.Iterator):
    """
    Read SAS files in SAS7BDAT format.

    Parameters
    ----------
    path_or_buf : path name or buffer
        Name of SAS file or file-like object pointing to SAS file
        contents.
    index : column identifier, defaults to None
        Column to use as index.
    convert_dates : bool, defaults to True
        Attempt to convert dates to Pandas datetime values.  Note that
        some rarely used SAS date formats may be unsupported.
    blank_missing : bool, defaults to True
        Convert empty strings to missing values (SAS uses blanks to
        indicate missing character variables).
    chunksize : int, defaults to None
        Return SAS7BDATReader object for iterations, returns chunks
        with given number of lines.
    encoding : string, defaults to None
        String encoding.
    convert_text : bool, defaults to True
        If False, text variables are left as raw bytes.
    convert_header_text : bool, defaults to True
        If False, header text, including column names, are left as raw
        bytes.
    """

    _int_length: int
    _cached_page: bytes | None

    def __init__(
        self,
        path_or_buf,
        index=None,
        convert_dates=True,
        blank_missing=True,
        chunksize=None,
        encoding=None,
        convert_text=True,
        convert_header_text=True,
    ):

        self.index = index
        self.convert_dates = convert_dates
        self.blank_missing = blank_missing
        self.chunksize = chunksize
        self.encoding = encoding
        self.convert_text = convert_text
        self.convert_header_text = convert_header_text

        self.default_encoding = "latin-1"
        self.compression = b""
        self.column_names_strings = []
        self.column_names = []
        self.column_formats = []
        self.columns = []

        self._current_page_data_subheader_pointers = []
        self._cached_page = None
        self._column_data_lengths = []
        self._column_data_offsets = []
        self._column_types = []

        self._current_row_in_file_index = 0
        self._current_row_on_page_index = 0
        self._current_row_in_file_index = 0

        self.handles = get_handle(path_or_buf, "rb", is_text=False)

        self._path_or_buf = cast(IO[Any], self.handles.handle)

        try:
            self._get_properties()
            self._parse_metadata()
        except Exception:
            self.close()
            raise

    def column_data_lengths(self) -> np.ndarray:
        """Return a numpy int64 array of the column data lengths"""
        return np.asarray(self._column_data_lengths, dtype=np.int64)

    def column_data_offsets(self) -> np.ndarray:
        """Return a numpy int64 array of the column offsets"""
        return np.asarray(self._column_data_offsets, dtype=np.int64)

    def column_types(self) -> np.ndarray:
        """
        Returns a numpy character array of the column types:
           s (string) or d (double)
        """
        return np.asarray(self._column_types, dtype=np.dtype("S1"))

    def close(self) -> None:
        self.handles.close()

    def _get_properties(self) -> None:

        # Check magic number
        self._path_or_buf.seek(0)
        self._cached_page = cast(bytes, self._path_or_buf.read(288))
        if self._cached_page[0 : len(const.magic)] != const.magic:
            raise ValueError("magic number mismatch (not a SAS file?)")

        # Get alignment information
        align1, align2 = 0, 0
        buf = self._read_bytes(const.align_1_offset, const.align_1_length)
        if buf == const.u64_byte_checker_value:
            align2 = const.align_2_value
            self.U64 = True
            self._int_length = 8
            self._page_bit_offset = const.page_bit_offset_x64
            self._subheader_pointer_length = const.subheader_pointer_length_x64
        else:
            self.U64 = False
            self._page_bit_offset = const.page_bit_offset_x86
            self._subheader_pointer_length = const.subheader_pointer_length_x86
            self._int_length = 4
        buf = self._read_bytes(const.align_2_offset, const.align_2_length)
        if buf == const.align_1_checker_value:
            align1 = const.align_2_value
        total_align = align1 + align2

        # Get endianness information
        buf = self._read_bytes(const.endianness_offset, const.endianness_length)
        if buf == b"\x01":
            self.byte_order = "<"
        else:
            self.byte_order = ">"

        # Get encoding information
        buf = self._read_bytes(const.encoding_offset, const.encoding_length)[0]
        if buf in const.encoding_names:
            self.file_encoding = const.encoding_names[buf]
        else:
            self.file_encoding = f"unknown (code={buf})"

        # Get platform information
        buf = self._read_bytes(const.platform_offset, const.platform_length)
        if buf == b"1":
            self.platform = "unix"
        elif buf == b"2":
            self.platform = "windows"
        else:
            self.platform = "unknown"

        buf = self._read_bytes(const.dataset_offset, const.dataset_length)
        self.name = buf.rstrip(b"\x00 ")
        if self.convert_header_text:
            self.name = self.name.decode(self.encoding or self.default_encoding)

        buf = self._read_bytes(const.file_type_offset, const.file_type_length)
        self.file_type = buf.rstrip(b"\x00 ")
        if self.convert_header_text:
            self.file_type = self.file_type.decode(
                self.encoding or self.default_encoding
            )

        # Timestamp is epoch 01/01/1960
        epoch = datetime(1960, 1, 1)
        x = self._read_float(
            const.date_created_offset + align1, const.date_created_length
        )
        self.date_created = epoch + pd.to_timedelta(x, unit="s")
        x = self._read_float(
            const.date_modified_offset + align1, const.date_modified_length
        )
        self.date_modified = epoch + pd.to_timedelta(x, unit="s")

        self.header_length = self._read_int(
            const.header_size_offset + align1, const.header_size_length
        )

        # Read the rest of the header into cached_page.
        buf = cast(bytes, self._path_or_buf.read(self.header_length - 288))
        self._cached_page += buf
        # error: Argument 1 to "len" has incompatible type "Optional[bytes]";
        #  expected "Sized"
        if len(self._cached_page) != self.header_length:  # type: ignore[arg-type]
            raise ValueError("The SAS7BDAT file appears to be truncated.")

        self._page_length = self._read_int(
            const.page_size_offset + align1, const.page_size_length
        )
        self._page_count = self._read_int(
            const.page_count_offset + align1, const.page_count_length
        )

        buf = self._read_bytes(
            const.sas_release_offset + total_align, const.sas_release_length
        )
        self.sas_release = buf.rstrip(b"\x00 ")
        if self.convert_header_text:
            self.sas_release = self.sas_release.decode(
                self.encoding or self.default_encoding
            )

        buf = self._read_bytes(
            const.sas_server_type_offset + total_align, const.sas_server_type_length
        )
        self.server_type = buf.rstrip(b"\x00 ")
        if self.convert_header_text:
            self.server_type = self.server_type.decode(
                self.encoding or self.default_encoding
            )

        buf = self._read_bytes(
            const.os_version_number_offset + total_align, const.os_version_number_length
        )
        self.os_version = buf.rstrip(b"\x00 ")
        if self.convert_header_text:
            self.os_version = self.os_version.decode(
                self.encoding or self.default_encoding
            )

        buf = self._read_bytes(const.os_name_offset + total_align, const.os_name_length)
        buf = buf.rstrip(b"\x00 ")
        if len(buf) > 0:
            self.os_name = buf.decode(self.encoding or self.default_encoding)
        else:
            buf = self._read_bytes(
                const.os_maker_offset + total_align, const.os_maker_length
            )
            self.os_name = buf.rstrip(b"\x00 ")
            if self.convert_header_text:
                self.os_name = self.os_name.decode(
                    self.encoding or self.default_encoding
                )

    def __next__(self):
        da = self.read(nrows=self.chunksize or 1)
        if da is None:
            self.close()
            raise StopIteration
        return da

    # Read a single float of the given width (4 or 8).
    def _read_float(self, offset: int, width: int):
        if width not in (4, 8):
            self.close()
            raise ValueError("invalid float width")
        buf = self._read_bytes(offset, width)
        fd = "f" if width == 4 else "d"
        return struct.unpack(self.byte_order + fd, buf)[0]

    # Read a single signed integer of the given width (1, 2, 4 or 8).
    def _read_int(self, offset: int, width: int) -> int:
        if width not in (1, 2, 4, 8):
            self.close()
            raise ValueError("invalid int width")
        buf = self._read_bytes(offset, width)
        it = {1: "b", 2: "h", 4: "l", 8: "q"}[width]
        iv = struct.unpack(self.byte_order + it, buf)[0]
        return iv

    def _read_bytes(self, offset: int, length: int):
        if self._cached_page is None:
            self._path_or_buf.seek(offset)
            buf = self._path_or_buf.read(length)
            if len(buf) < length:
                self.close()
                msg = f"Unable to read {length:d} bytes from file position {offset:d}."
                raise ValueError(msg)
            return buf
        else:
            if offset + length > len(self._cached_page):
                self.close()
                raise ValueError("The cached page is too small.")
            return self._cached_page[offset : offset + length]

    def _parse_metadata(self) -> None:
        done = False
        while not done:
            self._cached_page = cast(bytes, self._path_or_buf.read(self._page_length))
            if len(self._cached_page) <= 0:
                break
            if len(self._cached_page) != self._page_length:
                raise ValueError("Failed to read a meta data page from the SAS file.")
            done = self._process_page_meta()

    def _process_page_meta(self) -> bool:
        self._read_page_header()
        pt = [const.page_meta_type, const.page_amd_type] + const.page_mix_types
        if self._current_page_type in pt:
            self._process_page_metadata()
        is_data_page = self._current_page_type & const.page_data_type
        is_mix_page = self._current_page_type in const.page_mix_types
        return bool(
            is_data_page
            or is_mix_page
            or self._current_page_data_subheader_pointers != []
        )

    def _read_page_header(self):
        bit_offset = self._page_bit_offset
        tx = const.page_type_offset + bit_offset
        self._current_page_type = self._read_int(tx, const.page_type_length)
        tx = const.block_count_offset + bit_offset
        self._current_page_block_count = self._read_int(tx, const.block_count_length)
        tx = const.subheader_count_offset + bit_offset
        self._current_page_subheaders_count = self._read_int(
            tx, const.subheader_count_length
        )

    def _process_page_metadata(self) -> None:
        bit_offset = self._page_bit_offset

        for i in range(self._current_page_subheaders_count):
            pointer = self._process_subheader_pointers(
                const.subheader_pointers_offset + bit_offset, i
            )
            if pointer.length == 0:
                continue
            if pointer.compression == const.truncated_subheader_id:
                continue
            subheader_signature = self._read_subheader_signature(pointer.offset)
            subheader_index = self._get_subheader_index(
                subheader_signature, pointer.compression, pointer.ptype
            )
            self._process_subheader(subheader_index, pointer)

    def _get_subheader_index(self, signature: bytes, compression, ptype) -> int:
        # TODO: return here could be made an enum
        index = const.subheader_signature_to_index.get(signature)
        if index is None:
            f1 = (compression == const.compressed_subheader_id) or (compression == 0)
            f2 = ptype == const.compressed_subheader_type
            if (self.compression != b"") and f1 and f2:
                index = const.SASIndex.data_subheader_index
            else:
                self.close()
                raise ValueError("Unknown subheader signature")
        return index

    def _process_subheader_pointers(
        self, offset: int, subheader_pointer_index: int
    ) -> _SubheaderPointer:

        subheader_pointer_length = self._subheader_pointer_length
        total_offset = offset + subheader_pointer_length * subheader_pointer_index

        subheader_offset = self._read_int(total_offset, self._int_length)
        total_offset += self._int_length

        subheader_length = self._read_int(total_offset, self._int_length)
        total_offset += self._int_length

        subheader_compression = self._read_int(total_offset, 1)
        total_offset += 1

        subheader_type = self._read_int(total_offset, 1)

        x = _SubheaderPointer(
            subheader_offset, subheader_length, subheader_compression, subheader_type
        )

        return x

    def _read_subheader_signature(self, offset: int) -> bytes:
        subheader_signature = self._read_bytes(offset, self._int_length)
        return subheader_signature

    def _process_subheader(
        self, subheader_index: int, pointer: _SubheaderPointer
    ) -> None:
        offset = pointer.offset
        length = pointer.length

        if subheader_index == const.SASIndex.row_size_index:
            processor = self._process_rowsize_subheader
        elif subheader_index == const.SASIndex.column_size_index:
            processor = self._process_columnsize_subheader
        elif subheader_index == const.SASIndex.column_text_index:
            processor = self._process_columntext_subheader
        elif subheader_index == const.SASIndex.column_name_index:
            processor = self._process_columnname_subheader
        elif subheader_index == const.SASIndex.column_attributes_index:
            processor = self._process_columnattributes_subheader
        elif subheader_index == const.SASIndex.format_and_label_index:
            processor = self._process_format_subheader
        elif subheader_index == const.SASIndex.column_list_index:
            processor = self._process_columnlist_subheader
        elif subheader_index == const.SASIndex.subheader_counts_index:
            processor = self._process_subheader_counts
        elif subheader_index == const.SASIndex.data_subheader_index:
            self._current_page_data_subheader_pointers.append(pointer)
            return
        else:
            raise ValueError("unknown subheader index")

        processor(offset, length)

    def _process_rowsize_subheader(self, offset: int, length: int) -> None:

        int_len = self._int_length
        lcs_offset = offset
        lcp_offset = offset
        if self.U64:
            lcs_offset += 682
            lcp_offset += 706
        else:
            lcs_offset += 354
            lcp_offset += 378

        self.row_length = self._read_int(
            offset + const.row_length_offset_multiplier * int_len, int_len
        )
        self.row_count = self._read_int(
            offset + const.row_count_offset_multiplier * int_len, int_len
        )
        self.col_count_p1 = self._read_int(
            offset + const.col_count_p1_multiplier * int_len, int_len
        )
        self.col_count_p2 = self._read_int(
            offset + const.col_count_p2_multiplier * int_len, int_len
        )
        mx = const.row_count_on_mix_page_offset_multiplier * int_len
        self._mix_page_row_count = self._read_int(offset + mx, int_len)
        self._lcs = self._read_int(lcs_offset, 2)
        self._lcp = self._read_int(lcp_offset, 2)

    def _process_columnsize_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        self.column_count = self._read_int(offset, int_len)
        if self.col_count_p1 + self.col_count_p2 != self.column_count:
            print(
                f"Warning: column count mismatch ({self.col_count_p1} + "
                f"{self.col_count_p2} != {self.column_count})\n"
            )

    # Unknown purpose
    def _process_subheader_counts(self, offset: int, length: int) -> None:
        pass

    def _process_columntext_subheader(self, offset: int, length: int) -> None:

        offset += self._int_length
        text_block_size = self._read_int(offset, const.text_block_size_length)

        buf = self._read_bytes(offset, text_block_size)
        cname_raw = buf[0:text_block_size].rstrip(b"\x00 ")
        cname = cname_raw
        if self.convert_header_text:
            cname = cname.decode(self.encoding or self.default_encoding)
        self.column_names_strings.append(cname)

        if len(self.column_names_strings) == 1:
            compression_literal = b""
            for cl in const.compression_literals:
                if cl in cname_raw:
                    compression_literal = cl
            self.compression = compression_literal
            offset -= self._int_length

            offset1 = offset + 16
            if self.U64:
                offset1 += 4

            buf = self._read_bytes(offset1, self._lcp)
            compression_literal = buf.rstrip(b"\x00")
            if compression_literal == b"":
                self._lcs = 0
                offset1 = offset + 32
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcp)
                self.creator_proc = buf[0 : self._lcp]
            elif compression_literal == const.rle_compression:
                offset1 = offset + 40
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcp)
                self.creator_proc = buf[0 : self._lcp]
            elif self._lcs > 0:
                self._lcp = 0
                offset1 = offset + 16
                if self.U64:
                    offset1 += 4
                buf = self._read_bytes(offset1, self._lcs)
                self.creator_proc = buf[0 : self._lcp]
            if self.convert_header_text:
                if hasattr(self, "creator_proc"):
                    self.creator_proc = self.creator_proc.decode(
                        self.encoding or self.default_encoding
                    )

    def _process_columnname_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        offset += int_len
        column_name_pointers_count = (length - 2 * int_len - 12) // 8
        for i in range(column_name_pointers_count):
            text_subheader = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_text_subheader_offset
            )
            col_name_offset = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_offset_offset
            )
            col_name_length = (
                offset
                + const.column_name_pointer_length * (i + 1)
                + const.column_name_length_offset
            )

            idx = self._read_int(
                text_subheader, const.column_name_text_subheader_length
            )
            col_offset = self._read_int(
                col_name_offset, const.column_name_offset_length
            )
            col_len = self._read_int(col_name_length, const.column_name_length_length)

            name_str = self.column_names_strings[idx]
            self.column_names.append(name_str[col_offset : col_offset + col_len])

    def _process_columnattributes_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        column_attributes_vectors_count = (length - 2 * int_len - 12) // (int_len + 8)
        for i in range(column_attributes_vectors_count):
            col_data_offset = (
                offset + int_len + const.column_data_offset_offset + i * (int_len + 8)
            )
            col_data_len = (
                offset
                + 2 * int_len
                + const.column_data_length_offset
                + i * (int_len + 8)
            )
            col_types = (
                offset + 2 * int_len + const.column_type_offset + i * (int_len + 8)
            )

            x = self._read_int(col_data_offset, int_len)
            self._column_data_offsets.append(x)

            x = self._read_int(col_data_len, const.column_data_length_length)
            self._column_data_lengths.append(x)

            x = self._read_int(col_types, const.column_type_length)
            self._column_types.append(b"d" if x == 1 else b"s")

    def _process_columnlist_subheader(self, offset: int, length: int) -> None:
        # unknown purpose
        pass

    def _process_format_subheader(self, offset: int, length: int) -> None:
        int_len = self._int_length
        text_subheader_format = (
            offset + const.column_format_text_subheader_index_offset + 3 * int_len
        )
        col_format_offset = offset + const.column_format_offset_offset + 3 * int_len
        col_format_len = offset + const.column_format_length_offset + 3 * int_len
        text_subheader_label = (
            offset + const.column_label_text_subheader_index_offset + 3 * int_len
        )
        col_label_offset = offset + const.column_label_offset_offset + 3 * int_len
        col_label_len = offset + const.column_label_length_offset + 3 * int_len

        x = self._read_int(
            text_subheader_format, const.column_format_text_subheader_index_length
        )
        format_idx = min(x, len(self.column_names_strings) - 1)

        format_start = self._read_int(
            col_format_offset, const.column_format_offset_length
        )
        format_len = self._read_int(col_format_len, const.column_format_length_length)

        label_idx = self._read_int(
            text_subheader_label, const.column_label_text_subheader_index_length
        )
        label_idx = min(label_idx, len(self.column_names_strings) - 1)

        label_start = self._read_int(col_label_offset, const.column_label_offset_length)
        label_len = self._read_int(col_label_len, const.column_label_length_length)

        label_names = self.column_names_strings[label_idx]
        column_label = label_names[label_start : label_start + label_len]
        format_names = self.column_names_strings[format_idx]
        column_format = format_names[format_start : format_start + format_len]
        current_column_number = len(self.columns)

        col = _Column(
            current_column_number,
            self.column_names[current_column_number],
            column_label,
            column_format,
            self._column_types[current_column_number],
            self._column_data_lengths[current_column_number],
        )

        self.column_formats.append(column_format)
        self.columns.append(col)

    def read(self, nrows: int | None = None) -> DataFrame | None:

        if (nrows is None) and (self.chunksize is not None):
            nrows = self.chunksize
        elif nrows is None:
            nrows = self.row_count

        if len(self._column_types) == 0:
            self.close()
            raise EmptyDataError("No columns to parse from file")

        if self._current_row_in_file_index >= self.row_count:
            return None

        m = self.row_count - self._current_row_in_file_index
        if nrows > m:
            nrows = m

        nd = self._column_types.count(b"d")
        ns = self._column_types.count(b"s")

        self._string_chunk = np.empty((ns, nrows), dtype=object)
        self._byte_chunk = np.zeros((nd, 8 * nrows), dtype=np.uint8)

        self._current_row_in_chunk_index = 0
        p = Parser(self)
        p.read(nrows)

        rslt = self._chunk_to_dataframe()
        if self.index is not None:
            rslt = rslt.set_index(self.index)

        return rslt

    def _read_next_page(self):
        self._current_page_data_subheader_pointers = []
        self._cached_page = cast(bytes, self._path_or_buf.read(self._page_length))
        if len(self._cached_page) <= 0:
            return True
        elif len(self._cached_page) != self._page_length:
            self.close()
            msg = (
                "failed to read complete page from file (read "
                f"{len(self._cached_page):d} of {self._page_length:d} bytes)"
            )
            raise ValueError(msg)

        self._read_page_header()
        page_type = self._current_page_type
        if page_type == const.page_meta_type:
            self._process_page_metadata()

        is_data_page = page_type & const.page_data_type
        pt = [const.page_meta_type] + const.page_mix_types
        if not is_data_page and self._current_page_type not in pt:
            return self._read_next_page()

        return False

    def _chunk_to_dataframe(self) -> DataFrame:

        n = self._current_row_in_chunk_index
        m = self._current_row_in_file_index
        ix = range(m - n, m)
        rslt = DataFrame(index=ix)

        js, jb = 0, 0
        for j in range(self.column_count):

            name = self.column_names[j]

            if self._column_types[j] == b"d":
                rslt[name] = self._byte_chunk[jb, :].view(dtype=self.byte_order + "d")
                rslt[name] = np.asarray(rslt[name], dtype=np.float64)
                if self.convert_dates:
                    if self.column_formats[j] in const.sas_date_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "d")
                    elif self.column_formats[j] in const.sas_datetime_formats:
                        rslt[name] = _convert_datetimes(rslt[name], "s")
                jb += 1
            elif self._column_types[j] == b"s":
                rslt[name] = self._string_chunk[js, :]
                if self.convert_text and (self.encoding is not None):
                    rslt[name] = rslt[name].str.decode(
                        self.encoding or self.default_encoding
                    )
                if self.blank_missing:
                    ii = rslt[name].str.len() == 0
                    rslt.loc[ii, name] = np.nan
                js += 1
            else:
                self.close()
                raise ValueError(f"unknown column type {self._column_types[j]}")

        return rslt
