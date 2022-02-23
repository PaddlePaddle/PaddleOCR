# cython: profile=False
# cython: boundscheck=False, initializedcheck=False
from cython import Py_ssize_t
import numpy as np

import pandas.io.sas.sas_constants as const

ctypedef signed long long   int64_t
ctypedef unsigned char      uint8_t
ctypedef unsigned short     uint16_t

# rle_decompress decompresses data using a Run Length Encoding
# algorithm.  It is partially documented here:
#
# https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf
cdef const uint8_t[:] rle_decompress(int result_length, const uint8_t[:] inbuff):

    cdef:
        uint8_t control_byte, x
        uint8_t[:] result = np.zeros(result_length, np.uint8)
        int rpos = 0
        int i, nbytes, end_of_first_byte
        Py_ssize_t ipos = 0, length = len(inbuff)

    while ipos < length:
        control_byte = inbuff[ipos] & 0xF0
        end_of_first_byte = <int>(inbuff[ipos] & 0x0F)
        ipos += 1

        if control_byte == 0x00:
            if end_of_first_byte != 0:
                raise ValueError("Unexpected non-zero end_of_first_byte")
            nbytes = <int>(inbuff[ipos]) + 64
            ipos += 1
            for _ in range(nbytes):
                result[rpos] = inbuff[ipos]
                rpos += 1
                ipos += 1
        elif control_byte == 0x40:
            # not documented
            nbytes = end_of_first_byte * 16
            nbytes += <int>(inbuff[ipos])
            ipos += 1
            for _ in range(nbytes):
                result[rpos] = inbuff[ipos]
                rpos += 1
            ipos += 1
        elif control_byte == 0x60:
            nbytes = end_of_first_byte * 256 + <int>(inbuff[ipos]) + 17
            ipos += 1
            for _ in range(nbytes):
                result[rpos] = 0x20
                rpos += 1
        elif control_byte == 0x70:
            nbytes = end_of_first_byte * 256 + <int>(inbuff[ipos]) + 17
            ipos += 1
            for _ in range(nbytes):
                result[rpos] = 0x00
                rpos += 1
        elif control_byte == 0x80:
            nbytes = end_of_first_byte + 1
            for i in range(nbytes):
                result[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0x90:
            nbytes = end_of_first_byte + 17
            for i in range(nbytes):
                result[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xA0:
            nbytes = end_of_first_byte + 33
            for i in range(nbytes):
                result[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xB0:
            nbytes = end_of_first_byte + 49
            for i in range(nbytes):
                result[rpos] = inbuff[ipos + i]
                rpos += 1
            ipos += nbytes
        elif control_byte == 0xC0:
            nbytes = end_of_first_byte + 3
            x = inbuff[ipos]
            ipos += 1
            for _ in range(nbytes):
                result[rpos] = x
                rpos += 1
        elif control_byte == 0xD0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                result[rpos] = 0x40
                rpos += 1
        elif control_byte == 0xE0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                result[rpos] = 0x20
                rpos += 1
        elif control_byte == 0xF0:
            nbytes = end_of_first_byte + 2
            for _ in range(nbytes):
                result[rpos] = 0x00
                rpos += 1
        else:
            raise ValueError(f"unknown control byte: {control_byte}")

    # In py37 cython/clang sees `len(outbuff)` as size_t and not Py_ssize_t
    if <Py_ssize_t>len(result) != <Py_ssize_t>result_length:
        raise ValueError(f"RLE: {len(result)} != {result_length}")

    return np.asarray(result)


# rdc_decompress decompresses data using the Ross Data Compression algorithm:
#
# http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
cdef const uint8_t[:] rdc_decompress(int result_length, const uint8_t[:] inbuff):

    cdef:
        uint8_t cmd
        uint16_t ctrl_bits = 0, ctrl_mask = 0, ofs, cnt
        int rpos = 0, k
        uint8_t[:] outbuff = np.zeros(result_length, dtype=np.uint8)
        Py_ssize_t ipos = 0, length = len(inbuff)

    ii = -1

    while ipos < length:
        ii += 1
        ctrl_mask = ctrl_mask >> 1
        if ctrl_mask == 0:
            ctrl_bits = ((<uint16_t>inbuff[ipos] << 8) +
                         <uint16_t>inbuff[ipos + 1])
            ipos += 2
            ctrl_mask = 0x8000

        if ctrl_bits & ctrl_mask == 0:
            outbuff[rpos] = inbuff[ipos]
            ipos += 1
            rpos += 1
            continue

        cmd = (inbuff[ipos] >> 4) & 0x0F
        cnt = <uint16_t>(inbuff[ipos] & 0x0F)
        ipos += 1

        # short RLE
        if cmd == 0:
            cnt += 3
            for k in range(cnt):
                outbuff[rpos + k] = inbuff[ipos]
            rpos += cnt
            ipos += 1

        # long RLE
        elif cmd == 1:
            cnt += <uint16_t>inbuff[ipos] << 4
            cnt += 19
            ipos += 1
            for k in range(cnt):
                outbuff[rpos + k] = inbuff[ipos]
            rpos += cnt
            ipos += 1

        # long pattern
        elif cmd == 2:
            ofs = cnt + 3
            ofs += <uint16_t>inbuff[ipos] << 4
            ipos += 1
            cnt = <uint16_t>inbuff[ipos]
            ipos += 1
            cnt += 16
            for k in range(cnt):
                outbuff[rpos + k] = outbuff[rpos - <int>ofs + k]
            rpos += cnt

        # short pattern
        elif (cmd >= 3) & (cmd <= 15):
            ofs = cnt + 3
            ofs += <uint16_t>inbuff[ipos] << 4
            ipos += 1
            for k in range(cmd):
                outbuff[rpos + k] = outbuff[rpos - <int>ofs + k]
            rpos += cmd

        else:
            raise ValueError("unknown RDC command")

    # In py37 cython/clang sees `len(outbuff)` as size_t and not Py_ssize_t
    if <Py_ssize_t>len(outbuff) != <Py_ssize_t>result_length:
        raise ValueError(f"RDC: {len(outbuff)} != {result_length}\n")

    return np.asarray(outbuff)


cdef enum ColumnTypes:
    column_type_decimal = 1
    column_type_string = 2


# type the page_data types
cdef:
    int page_meta_type = const.page_meta_type
    int page_mix_types_0 = const.page_mix_types[0]
    int page_mix_types_1 = const.page_mix_types[1]
    int page_data_type = const.page_data_type
    int subheader_pointers_offset = const.subheader_pointers_offset


cdef class Parser:

    cdef:
        int column_count
        int64_t[:] lengths
        int64_t[:] offsets
        int64_t[:] column_types
        uint8_t[:, :] byte_chunk
        object[:, :] string_chunk
        char *cached_page
        int current_row_on_page_index
        int current_page_block_count
        int current_page_data_subheader_pointers_len
        int current_page_subheaders_count
        int current_row_in_chunk_index
        int current_row_in_file_index
        int header_length
        int row_length
        int bit_offset
        int subheader_pointer_length
        int current_page_type
        bint is_little_endian
        const uint8_t[:] (*decompress)(int result_length, const uint8_t[:] inbuff)
        object parser

    def __init__(self, object parser):
        cdef:
            int j
            char[:] column_types

        self.parser = parser
        self.header_length = self.parser.header_length
        self.column_count = parser.column_count
        self.lengths = parser.column_data_lengths()
        self.offsets = parser.column_data_offsets()
        self.byte_chunk = parser._byte_chunk
        self.string_chunk = parser._string_chunk
        self.row_length = parser.row_length
        self.bit_offset = self.parser._page_bit_offset
        self.subheader_pointer_length = self.parser._subheader_pointer_length
        self.is_little_endian = parser.byte_order == "<"
        self.column_types = np.empty(self.column_count, dtype='int64')

        # page indicators
        self.update_next_page()

        column_types = parser.column_types()

        # map column types
        for j in range(self.column_count):
            if column_types[j] == b'd':
                self.column_types[j] = column_type_decimal
            elif column_types[j] == b's':
                self.column_types[j] = column_type_string
            else:
                raise ValueError(f"unknown column type: {self.parser.columns[j].ctype}")

        # compression
        if parser.compression == const.rle_compression:
            self.decompress = rle_decompress
        elif parser.compression == const.rdc_compression:
            self.decompress = rdc_decompress
        else:
            self.decompress = NULL

        # update to current state of the parser
        self.current_row_in_chunk_index = parser._current_row_in_chunk_index
        self.current_row_in_file_index = parser._current_row_in_file_index
        self.current_row_on_page_index = parser._current_row_on_page_index

    def read(self, int nrows):
        cdef:
            bint done
            int i

        for _ in range(nrows):
            done = self.readline()
            if done:
                break

        # update the parser
        self.parser._current_row_on_page_index = self.current_row_on_page_index
        self.parser._current_row_in_chunk_index = self.current_row_in_chunk_index
        self.parser._current_row_in_file_index = self.current_row_in_file_index

    cdef bint read_next_page(self):
        cdef done

        done = self.parser._read_next_page()
        if done:
            self.cached_page = NULL
        else:
            self.update_next_page()
        return done

    cdef update_next_page(self):
        # update data for the current page

        self.cached_page = <char *>self.parser._cached_page
        self.current_row_on_page_index = 0
        self.current_page_type = self.parser._current_page_type
        self.current_page_block_count = self.parser._current_page_block_count
        self.current_page_data_subheader_pointers_len = len(
            self.parser._current_page_data_subheader_pointers
        )
        self.current_page_subheaders_count = self.parser._current_page_subheaders_count

    cdef readline(self):

        cdef:
            int offset, bit_offset, align_correction
            int subheader_pointer_length, mn
            bint done, flag

        bit_offset = self.bit_offset
        subheader_pointer_length = self.subheader_pointer_length

        # If there is no page, go to the end of the header and read a page.
        if self.cached_page == NULL:
            self.parser._path_or_buf.seek(self.header_length)
            done = self.read_next_page()
            if done:
                return True

        # Loop until a data row is read
        while True:
            if self.current_page_type == page_meta_type:
                flag = self.current_row_on_page_index >=\
                    self.current_page_data_subheader_pointers_len
                if flag:
                    done = self.read_next_page()
                    if done:
                        return True
                    continue
                current_subheader_pointer = (
                    self.parser._current_page_data_subheader_pointers[
                        self.current_row_on_page_index])
                self.process_byte_array_with_data(
                    current_subheader_pointer.offset,
                    current_subheader_pointer.length)
                return False
            elif (self.current_page_type == page_mix_types_0 or
                    self.current_page_type == page_mix_types_1):
                align_correction = (
                    bit_offset
                    + subheader_pointers_offset
                    + self.current_page_subheaders_count * subheader_pointer_length
                )
                align_correction = align_correction % 8
                offset = bit_offset + align_correction
                offset += subheader_pointers_offset
                offset += self.current_page_subheaders_count * subheader_pointer_length
                offset += self.current_row_on_page_index * self.row_length
                self.process_byte_array_with_data(offset, self.row_length)
                mn = min(self.parser.row_count, self.parser._mix_page_row_count)
                if self.current_row_on_page_index == mn:
                    done = self.read_next_page()
                    if done:
                        return True
                return False
            elif self.current_page_type & page_data_type == page_data_type:
                self.process_byte_array_with_data(
                    bit_offset
                    + subheader_pointers_offset
                    + self.current_row_on_page_index * self.row_length,
                    self.row_length,
                )
                flag = self.current_row_on_page_index == self.current_page_block_count
                if flag:
                    done = self.read_next_page()
                    if done:
                        return True
                return False
            else:
                raise ValueError(f"unknown page type: {self.current_page_type}")

    cdef void process_byte_array_with_data(self, int offset, int length):

        cdef:
            Py_ssize_t j
            int s, k, m, jb, js, current_row
            int64_t lngt, start, ct
            const uint8_t[:] source
            int64_t[:] column_types
            int64_t[:] lengths
            int64_t[:] offsets
            uint8_t[:, :] byte_chunk
            object[:, :] string_chunk

        source = np.frombuffer(
            self.cached_page[offset:offset + length], dtype=np.uint8)

        if self.decompress != NULL and (length < self.row_length):
            source = self.decompress(self.row_length, source)

        current_row = self.current_row_in_chunk_index
        column_types = self.column_types
        lengths = self.lengths
        offsets = self.offsets
        byte_chunk = self.byte_chunk
        string_chunk = self.string_chunk
        s = 8 * self.current_row_in_chunk_index
        js = 0
        jb = 0
        for j in range(self.column_count):
            lngt = lengths[j]
            if lngt == 0:
                break
            start = offsets[j]
            ct = column_types[j]
            if ct == column_type_decimal:
                # decimal
                if self.is_little_endian:
                    m = s + 8 - lngt
                else:
                    m = s
                for k in range(lngt):
                    byte_chunk[jb, m + k] = source[start + k]
                jb += 1
            elif column_types[j] == column_type_string:
                # string
                string_chunk[js, current_row] = np.array(source[start:(
                    start + lngt)]).tobytes().rstrip(b"\x00 ")
                js += 1

        self.current_row_on_page_index += 1
        self.current_row_in_chunk_index += 1
        self.current_row_in_file_index += 1
