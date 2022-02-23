"""
Implementation of Harwell-Boeing read/write.

At the moment not the full Harwell-Boeing format is supported. Supported
features are:

    - assembled, non-symmetric, real matrices
    - integer for pointer/indices
    - exponential format for float values, and int format

"""
# TODO:
#   - Add more support (symmetric/complex matrices, non-assembled matrices ?)

# XXX: reading is reasonably efficient (>= 85 % is in numpy.fromstring), but
# takes a lot of memory. Being faster would require compiled code.
# write is not efficient. Although not a terribly exciting task,
# having reusable facilities to efficiently read/write fortran-formatted files
# would be useful outside this module.

import warnings

import numpy as np
from scipy.sparse import csc_matrix
from scipy.io.harwell_boeing._fortran_format_parser import \
        FortranFormatParser, IntFormat, ExpFormat

__all__ = ["MalformedHeader", "hb_read", "hb_write", "HBInfo", "HBFile",
           "HBMatrixType"]


class MalformedHeader(Exception):
    pass


class LineOverflow(Warning):
    pass


def _nbytes_full(fmt, nlines):
    """Return the number of bytes to read to get every full lines for the
    given parsed fortran format."""
    return (fmt.repeat * fmt.width + 1) * (nlines - 1)


class HBInfo:
    @classmethod
    def from_data(cls, m, title="Default title", key="0", mxtype=None, fmt=None):
        """Create a HBInfo instance from an existing sparse matrix.

        Parameters
        ----------
        m : sparse matrix
            the HBInfo instance will derive its parameters from m
        title : str
            Title to put in the HB header
        key : str
            Key
        mxtype : HBMatrixType
            type of the input matrix
        fmt : dict
            not implemented

        Returns
        -------
        hb_info : HBInfo instance
        """
        m = m.tocsc(copy=False)

        pointer = m.indptr
        indices = m.indices
        values = m.data

        nrows, ncols = m.shape
        nnon_zeros = m.nnz

        if fmt is None:
            # +1 because HB use one-based indexing (Fortran), and we will write
            # the indices /pointer as such
            pointer_fmt = IntFormat.from_number(np.max(pointer+1))
            indices_fmt = IntFormat.from_number(np.max(indices+1))

            if values.dtype.kind in np.typecodes["AllFloat"]:
                values_fmt = ExpFormat.from_number(-np.max(np.abs(values)))
            elif values.dtype.kind in np.typecodes["AllInteger"]:
                values_fmt = IntFormat.from_number(-np.max(np.abs(values)))
            else:
                raise NotImplementedError("type %s not implemented yet" % values.dtype.kind)
        else:
            raise NotImplementedError("fmt argument not supported yet.")

        if mxtype is None:
            if not np.isrealobj(values):
                raise ValueError("Complex values not supported yet")
            if values.dtype.kind in np.typecodes["AllInteger"]:
                tp = "integer"
            elif values.dtype.kind in np.typecodes["AllFloat"]:
                tp = "real"
            else:
                raise NotImplementedError("type %s for values not implemented"
                                          % values.dtype)
            mxtype = HBMatrixType(tp, "unsymmetric", "assembled")
        else:
            raise ValueError("mxtype argument not handled yet.")

        def _nlines(fmt, size):
            nlines = size // fmt.repeat
            if nlines * fmt.repeat != size:
                nlines += 1
            return nlines

        pointer_nlines = _nlines(pointer_fmt, pointer.size)
        indices_nlines = _nlines(indices_fmt, indices.size)
        values_nlines = _nlines(values_fmt, values.size)

        total_nlines = pointer_nlines + indices_nlines + values_nlines

        return cls(title, key,
            total_nlines, pointer_nlines, indices_nlines, values_nlines,
            mxtype, nrows, ncols, nnon_zeros,
            pointer_fmt.fortran_format, indices_fmt.fortran_format,
            values_fmt.fortran_format)

    @classmethod
    def from_file(cls, fid):
        """Create a HBInfo instance from a file object containing a matrix in the
        HB format.

        Parameters
        ----------
        fid : file-like matrix
            File or file-like object containing a matrix in the HB format.

        Returns
        -------
        hb_info : HBInfo instance
        """
        # First line
        line = fid.readline().strip("\n")
        if not len(line) > 72:
            raise ValueError("Expected at least 72 characters for first line, "
                             "got: \n%s" % line)
        title = line[:72]
        key = line[72:]

        # Second line
        line = fid.readline().strip("\n")
        if not len(line.rstrip()) >= 56:
            raise ValueError("Expected at least 56 characters for second line, "
                             "got: \n%s" % line)
        total_nlines = _expect_int(line[:14])
        pointer_nlines = _expect_int(line[14:28])
        indices_nlines = _expect_int(line[28:42])
        values_nlines = _expect_int(line[42:56])

        rhs_nlines = line[56:72].strip()
        if rhs_nlines == '':
            rhs_nlines = 0
        else:
            rhs_nlines = _expect_int(rhs_nlines)
        if not rhs_nlines == 0:
            raise ValueError("Only files without right hand side supported for "
                             "now.")

        # Third line
        line = fid.readline().strip("\n")
        if not len(line) >= 70:
            raise ValueError("Expected at least 72 character for third line, got:\n"
                             "%s" % line)

        mxtype_s = line[:3].upper()
        if not len(mxtype_s) == 3:
            raise ValueError("mxtype expected to be 3 characters long")

        mxtype = HBMatrixType.from_fortran(mxtype_s)
        if mxtype.value_type not in ["real", "integer"]:
            raise ValueError("Only real or integer matrices supported for "
                             "now (detected %s)" % mxtype)
        if not mxtype.structure == "unsymmetric":
            raise ValueError("Only unsymmetric matrices supported for "
                             "now (detected %s)" % mxtype)
        if not mxtype.storage == "assembled":
            raise ValueError("Only assembled matrices supported for now")

        if not line[3:14] == " " * 11:
            raise ValueError("Malformed data for third line: %s" % line)

        nrows = _expect_int(line[14:28])
        ncols = _expect_int(line[28:42])
        nnon_zeros = _expect_int(line[42:56])
        nelementals = _expect_int(line[56:70])
        if not nelementals == 0:
            raise ValueError("Unexpected value %d for nltvl (last entry of line 3)"
                             % nelementals)

        # Fourth line
        line = fid.readline().strip("\n")

        ct = line.split()
        if not len(ct) == 3:
            raise ValueError("Expected 3 formats, got %s" % ct)

        return cls(title, key,
                   total_nlines, pointer_nlines, indices_nlines, values_nlines,
                   mxtype, nrows, ncols, nnon_zeros,
                   ct[0], ct[1], ct[2],
                   rhs_nlines, nelementals)

    def __init__(self, title, key,
            total_nlines, pointer_nlines, indices_nlines, values_nlines,
            mxtype, nrows, ncols, nnon_zeros,
            pointer_format_str, indices_format_str, values_format_str,
            right_hand_sides_nlines=0, nelementals=0):
        """Do not use this directly, but the class ctrs (from_* functions)."""
        self.title = title
        self.key = key
        if title is None:
            title = "No Title"
        if len(title) > 72:
            raise ValueError("title cannot be > 72 characters")

        if key is None:
            key = "|No Key"
        if len(key) > 8:
            warnings.warn("key is > 8 characters (key is %s)" % key, LineOverflow)

        self.total_nlines = total_nlines
        self.pointer_nlines = pointer_nlines
        self.indices_nlines = indices_nlines
        self.values_nlines = values_nlines

        parser = FortranFormatParser()
        pointer_format = parser.parse(pointer_format_str)
        if not isinstance(pointer_format, IntFormat):
            raise ValueError("Expected int format for pointer format, got %s"
                             % pointer_format)

        indices_format = parser.parse(indices_format_str)
        if not isinstance(indices_format, IntFormat):
            raise ValueError("Expected int format for indices format, got %s" %
                             indices_format)

        values_format = parser.parse(values_format_str)
        if isinstance(values_format, ExpFormat):
            if mxtype.value_type not in ["real", "complex"]:
                raise ValueError("Inconsistency between matrix type %s and "
                                 "value type %s" % (mxtype, values_format))
            values_dtype = np.float64
        elif isinstance(values_format, IntFormat):
            if mxtype.value_type not in ["integer"]:
                raise ValueError("Inconsistency between matrix type %s and "
                                 "value type %s" % (mxtype, values_format))
            # XXX: fortran int -> dtype association ?
            values_dtype = int
        else:
            raise ValueError("Unsupported format for values %r" % (values_format,))

        self.pointer_format = pointer_format
        self.indices_format = indices_format
        self.values_format = values_format

        self.pointer_dtype = np.int32
        self.indices_dtype = np.int32
        self.values_dtype = values_dtype

        self.pointer_nlines = pointer_nlines
        self.pointer_nbytes_full = _nbytes_full(pointer_format, pointer_nlines)

        self.indices_nlines = indices_nlines
        self.indices_nbytes_full = _nbytes_full(indices_format, indices_nlines)

        self.values_nlines = values_nlines
        self.values_nbytes_full = _nbytes_full(values_format, values_nlines)

        self.nrows = nrows
        self.ncols = ncols
        self.nnon_zeros = nnon_zeros
        self.nelementals = nelementals
        self.mxtype = mxtype

    def dump(self):
        """Gives the header corresponding to this instance as a string."""
        header = [self.title.ljust(72) + self.key.ljust(8)]

        header.append("%14d%14d%14d%14d" %
                      (self.total_nlines, self.pointer_nlines,
                       self.indices_nlines, self.values_nlines))
        header.append("%14s%14d%14d%14d%14d" %
                      (self.mxtype.fortran_format.ljust(14), self.nrows,
                       self.ncols, self.nnon_zeros, 0))

        pffmt = self.pointer_format.fortran_format
        iffmt = self.indices_format.fortran_format
        vffmt = self.values_format.fortran_format
        header.append("%16s%16s%20s" %
                      (pffmt.ljust(16), iffmt.ljust(16), vffmt.ljust(20)))
        return "\n".join(header)


def _expect_int(value, msg=None):
    try:
        return int(value)
    except ValueError as e:
        if msg is None:
            msg = "Expected an int, got %s"
        raise ValueError(msg % value) from e


def _read_hb_data(content, header):
    # XXX: look at a way to reduce memory here (big string creation)
    ptr_string = "".join([content.read(header.pointer_nbytes_full),
                           content.readline()])
    ptr = np.fromstring(ptr_string,
            dtype=int, sep=' ')

    ind_string = "".join([content.read(header.indices_nbytes_full),
                       content.readline()])
    ind = np.fromstring(ind_string,
            dtype=int, sep=' ')

    val_string = "".join([content.read(header.values_nbytes_full),
                          content.readline()])
    val = np.fromstring(val_string,
            dtype=header.values_dtype, sep=' ')

    try:
        return csc_matrix((val, ind-1, ptr-1),
                          shape=(header.nrows, header.ncols))
    except ValueError as e:
        raise e


def _write_data(m, fid, header):
    m = m.tocsc(copy=False)

    def write_array(f, ar, nlines, fmt):
        # ar_nlines is the number of full lines, n is the number of items per
        # line, ffmt the fortran format
        pyfmt = fmt.python_format
        pyfmt_full = pyfmt * fmt.repeat

        # for each array to write, we first write the full lines, and special
        # case for partial line
        full = ar[:(nlines - 1) * fmt.repeat]
        for row in full.reshape((nlines-1, fmt.repeat)):
            f.write(pyfmt_full % tuple(row) + "\n")
        nremain = ar.size - full.size
        if nremain > 0:
            f.write((pyfmt * nremain) % tuple(ar[ar.size - nremain:]) + "\n")

    fid.write(header.dump())
    fid.write("\n")
    # +1 is for Fortran one-based indexing
    write_array(fid, m.indptr+1, header.pointer_nlines,
                header.pointer_format)
    write_array(fid, m.indices+1, header.indices_nlines,
                header.indices_format)
    write_array(fid, m.data, header.values_nlines,
                header.values_format)


class HBMatrixType:
    """Class to hold the matrix type."""
    # q2f* translates qualified names to Fortran character
    _q2f_type = {
        "real": "R",
        "complex": "C",
        "pattern": "P",
        "integer": "I",
    }
    _q2f_structure = {
            "symmetric": "S",
            "unsymmetric": "U",
            "hermitian": "H",
            "skewsymmetric": "Z",
            "rectangular": "R"
    }
    _q2f_storage = {
        "assembled": "A",
        "elemental": "E",
    }

    _f2q_type = dict([(j, i) for i, j in _q2f_type.items()])
    _f2q_structure = dict([(j, i) for i, j in _q2f_structure.items()])
    _f2q_storage = dict([(j, i) for i, j in _q2f_storage.items()])

    @classmethod
    def from_fortran(cls, fmt):
        if not len(fmt) == 3:
            raise ValueError("Fortran format for matrix type should be 3 "
                             "characters long")
        try:
            value_type = cls._f2q_type[fmt[0]]
            structure = cls._f2q_structure[fmt[1]]
            storage = cls._f2q_storage[fmt[2]]
            return cls(value_type, structure, storage)
        except KeyError as e:
            raise ValueError("Unrecognized format %s" % fmt) from e

    def __init__(self, value_type, structure, storage="assembled"):
        self.value_type = value_type
        self.structure = structure
        self.storage = storage

        if value_type not in self._q2f_type:
            raise ValueError("Unrecognized type %s" % value_type)
        if structure not in self._q2f_structure:
            raise ValueError("Unrecognized structure %s" % structure)
        if storage not in self._q2f_storage:
            raise ValueError("Unrecognized storage %s" % storage)

    @property
    def fortran_format(self):
        return self._q2f_type[self.value_type] + \
               self._q2f_structure[self.structure] + \
               self._q2f_storage[self.storage]

    def __repr__(self):
        return "HBMatrixType(%s, %s, %s)" % \
               (self.value_type, self.structure, self.storage)


class HBFile:
    def __init__(self, file, hb_info=None):
        """Create a HBFile instance.

        Parameters
        ----------
        file : file-object
            StringIO work as well
        hb_info : HBInfo, optional
            Should be given as an argument for writing, in which case the file
            should be writable.
        """
        self._fid = file
        if hb_info is None:
            self._hb_info = HBInfo.from_file(file)
        else:
            #raise IOError("file %s is not writable, and hb_info "
            #              "was given." % file)
            self._hb_info = hb_info

    @property
    def title(self):
        return self._hb_info.title

    @property
    def key(self):
        return self._hb_info.key

    @property
    def type(self):
        return self._hb_info.mxtype.value_type

    @property
    def structure(self):
        return self._hb_info.mxtype.structure

    @property
    def storage(self):
        return self._hb_info.mxtype.storage

    def read_matrix(self):
        return _read_hb_data(self._fid, self._hb_info)

    def write_matrix(self, m):
        return _write_data(m, self._fid, self._hb_info)


def hb_read(path_or_open_file):
    """Read HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before reading.

    Returns
    -------
    data : scipy.sparse.csc_matrix instance
        The data read from the HB file as a sparse matrix.

    Notes
    -----
    At the moment not the full Harwell-Boeing format is supported. Supported
    features are:

        - assembled, non-symmetric, real matrices
        - integer for pointer/indices
        - exponential format for float values, and int format

    Examples
    --------
    We can read and write a harwell-boeing format file:

    >>> from scipy.io.harwell_boeing import hb_read, hb_write
    >>> from scipy.sparse import csr_matrix, eye
    >>> data = csr_matrix(eye(3))  # create a sparse matrix
    >>> hb_write("data.hb", data)  # write a hb file
    >>> print(hb_read("data.hb"))  # read a hb file
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0

    """
    def _get_matrix(fid):
        hb = HBFile(fid)
        return hb.read_matrix()

    if hasattr(path_or_open_file, 'read'):
        return _get_matrix(path_or_open_file)
    else:
        with open(path_or_open_file) as f:
            return _get_matrix(f)


def hb_write(path_or_open_file, m, hb_info=None):
    """Write HB-format file.

    Parameters
    ----------
    path_or_open_file : path-like or file-like
        If a file-like object, it is used as-is. Otherwise, it is opened
        before writing.
    m : sparse-matrix
        the sparse matrix to write
    hb_info : HBInfo
        contains the meta-data for write

    Returns
    -------
    None

    Notes
    -----
    At the moment not the full Harwell-Boeing format is supported. Supported
    features are:

        - assembled, non-symmetric, real matrices
        - integer for pointer/indices
        - exponential format for float values, and int format

    Examples
    --------
    We can read and write a harwell-boeing format file:

    >>> from scipy.io.harwell_boeing import hb_read, hb_write
    >>> from scipy.sparse import csr_matrix, eye
    >>> data = csr_matrix(eye(3))  # create a sparse matrix
    >>> hb_write("data.hb", data)  # write a hb file
    >>> print(hb_read("data.hb"))  # read a hb file
      (0, 0)	1.0
      (1, 1)	1.0
      (2, 2)	1.0

    """
    m = m.tocsc(copy=False)

    if hb_info is None:
        hb_info = HBInfo.from_data(m)

    def _set_matrix(fid):
        hb = HBFile(fid, hb_info)
        return hb.write_matrix(m)

    if hasattr(path_or_open_file, 'write'):
        return _set_matrix(path_or_open_file)
    else:
        with open(path_or_open_file, 'w') as f:
            return _set_matrix(f)
