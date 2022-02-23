''' Classes for read / write of matlab (TM) 5 files

The matfile specification last found here:

https://www.mathworks.com/access/helpdesk/help/pdf_doc/matlab/matfile_format.pdf

(as of December 5 2008)
'''
'''
=================================
 Note on functions and mat files
=================================

The document above does not give any hints as to the storage of matlab
function handles, or anonymous function handles. I had, therefore, to
guess the format of matlab arrays of ``mxFUNCTION_CLASS`` and
``mxOPAQUE_CLASS`` by looking at example mat files.

``mxFUNCTION_CLASS`` stores all types of matlab functions. It seems to
contain a struct matrix with a set pattern of fields. For anonymous
functions, a sub-fields of one of these fields seems to contain the
well-named ``mxOPAQUE_CLASS``. This seems to contain:

* array flags as for any matlab matrix
* 3 int8 strings
* a matrix

It seems that whenever the mat file contains a ``mxOPAQUE_CLASS``
instance, there is also an un-named matrix (name == '') at the end of
the mat file. I'll call this the ``__function_workspace__`` matrix.

When I saved two anonymous functions in a mat file, or appended another
anonymous function to the mat file, there was still only one
``__function_workspace__`` un-named matrix at the end, but larger than
that for a mat file with a single anonymous function, suggesting that
the workspaces for the two functions had been merged.

The ``__function_workspace__`` matrix appears to be of double class
(``mxCLASS_DOUBLE``), but stored as uint8, the memory for which is in
the format of a mini .mat file, without the first 124 bytes of the file
header (the description and the subsystem_offset), but with the version
U2 bytes, and the S2 endian test bytes. There follow 4 zero bytes,
presumably for 8 byte padding, and then a series of ``miMATRIX``
entries, as in a standard mat file. The ``miMATRIX`` entries appear to
be series of un-named (name == '') matrices, and may also contain arrays
of this same mini-mat format.

I guess that:

* saving an anonymous function back to a mat file will need the
  associated ``__function_workspace__`` matrix saved as well for the
  anonymous function to work correctly.
* appending to a mat file that has a ``__function_workspace__`` would
  involve first pulling off this workspace, appending, checking whether
  there were any more anonymous functions appended, and then somehow
  merging the relevant workspaces, and saving at the end of the mat
  file.

The mat files I was playing with are in ``tests/data``:

* sqr.mat
* parabola.mat
* some_functions.mat

See ``tests/test_mio.py:test_mio_funcs.py`` for the debugging
script I was working with.

'''

# Small fragments of current code adapted from matfile.py by Heiko
# Henkelmann; parts of the code for simplify_cells=True adapted from
# http://blog.nephics.com/2019/08/28/better-loadmat-for-scipy/.

import os
import time
import sys
import zlib

from io import BytesIO

import warnings

import numpy as np

import scipy.sparse

from .byteordercodes import native_code, swapped_code

from .miobase import (MatFileReader, docfiller, matdims, read_dtype,
                      arr_to_chars, arr_dtype_number, MatWriteError,
                      MatReadError, MatReadWarning)

# Reader object for matlab 5 format variables
from .mio5_utils import VarReader5

# Constants and helper objects
from .mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
                          NP_TO_MXTYPES, miCOMPRESSED, miMATRIX, miINT8,
                          miUTF8, miUINT32, mxCELL_CLASS, mxSTRUCT_CLASS,
                          mxOBJECT_CLASS, mxCHAR_CLASS, mxSPARSE_CLASS,
                          mxDOUBLE_CLASS, mclass_info, mat_struct)

from .streams import ZlibInputStream


def _has_struct(elem):
    """Determine if elem is an array and if first array item is a struct."""
    return (isinstance(elem, np.ndarray) and (elem.size > 0) and
            isinstance(elem[0], mat_struct))


def _inspect_cell_array(ndarray):
    """Construct lists from cell arrays (loaded as numpy ndarrays), recursing
    into items if they contain mat_struct objects."""
    elem_list = []
    for sub_elem in ndarray:
        if isinstance(sub_elem, mat_struct):
            elem_list.append(_matstruct_to_dict(sub_elem))
        elif _has_struct(sub_elem):
            elem_list.append(_inspect_cell_array(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list


def _matstruct_to_dict(matobj):
    """Construct nested dicts from mat_struct objects."""
    d = {}
    for f in matobj._fieldnames:
        elem = matobj.__dict__[f]
        if isinstance(elem, mat_struct):
            d[f] = _matstruct_to_dict(elem)
        elif _has_struct(elem):
            d[f] = _inspect_cell_array(elem)
        else:
            d[f] = elem
    return d


def _simplify_cells(d):
    """Convert mat objects in dict to nested dicts."""
    for key in d:
        if isinstance(d[key], mat_struct):
            d[key] = _matstruct_to_dict(d[key])
        elif _has_struct(d[key]):
            d[key] = _inspect_cell_array(d[key])
    return d


class MatFile5Reader(MatFileReader):
    ''' Reader for Mat 5 mat files
    Adds the following attribute to base class

    uint16_codec - char codec to use for uint16 char arrays
        (defaults to system default codec)

    Uses variable reader that has the following stardard interface (see
    abstract class in ``miobase``::

       __init__(self, file_reader)
       read_header(self)
       array_from_header(self)

    and added interface::

       set_stream(self, stream)
       read_full_tag(self)

    '''
    @docfiller
    def __init__(self,
                 mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 uint16_codec=None,
                 simplify_cells=False):
        '''Initializer for matlab 5 file format reader

    %(matstream_arg)s
    %(load_args)s
    %(struct_arg)s
    uint16_codec : {None, string}
        Set codec to use for uint16 char arrays (e.g., 'utf-8').
        Use system default codec if None
        '''
        super().__init__(
            mat_stream,
            byte_order,
            mat_dtype,
            squeeze_me,
            chars_as_strings,
            matlab_compatible,
            struct_as_record,
            verify_compressed_data_integrity,
            simplify_cells)
        # Set uint16 codec
        if not uint16_codec:
            uint16_codec = sys.getdefaultencoding()
        self.uint16_codec = uint16_codec
        # placeholders for readers - see initialize_read method
        self._file_reader = None
        self._matrix_reader = None

    def guess_byte_order(self):
        ''' Guess byte order.
        Sets stream pointer to 0 '''
        self.mat_stream.seek(126)
        mi = self.mat_stream.read(2)
        self.mat_stream.seek(0)
        return mi == b'IM' and '<' or '>'

    def read_file_header(self):
        ''' Read in mat 5 file header '''
        hdict = {}
        hdr_dtype = MDTYPES[self.byte_order]['dtypes']['file_header']
        hdr = read_dtype(self.mat_stream, hdr_dtype)
        hdict['__header__'] = hdr['description'].item().strip(b' \t\n\000')
        v_major = hdr['version'] >> 8
        v_minor = hdr['version'] & 0xFF
        hdict['__version__'] = '%d.%d' % (v_major, v_minor)
        return hdict

    def initialize_read(self):
        ''' Run when beginning read of variables

        Sets up readers from parameters in `self`
        '''
        # reader for top level stream. We need this extra top-level
        # reader because we use the matrix_reader object to contain
        # compressed matrices (so they have their own stream)
        self._file_reader = VarReader5(self)
        # reader for matrix streams
        self._matrix_reader = VarReader5(self)

    def read_var_header(self):
        ''' Read header, return header, next position

        Header has to define at least .name and .is_global

        Parameters
        ----------
        None

        Returns
        -------
        header : object
           object that can be passed to self.read_var_array, and that
           has attributes .name and .is_global
        next_position : int
           position in stream of next variable
        '''
        mdtype, byte_count = self._file_reader.read_full_tag()
        if not byte_count > 0:
            raise ValueError("Did not read any bytes")
        next_pos = self.mat_stream.tell() + byte_count
        if mdtype == miCOMPRESSED:
            # Make new stream from compressed data
            stream = ZlibInputStream(self.mat_stream, byte_count)
            self._matrix_reader.set_stream(stream)
            check_stream_limit = self.verify_compressed_data_integrity
            mdtype, byte_count = self._matrix_reader.read_full_tag()
        else:
            check_stream_limit = False
            self._matrix_reader.set_stream(self.mat_stream)
        if not mdtype == miMATRIX:
            raise TypeError('Expecting miMATRIX type here, got %d' % mdtype)
        header = self._matrix_reader.read_header(check_stream_limit)
        return header, next_pos

    def read_var_array(self, header, process=True):
        ''' Read array, given `header`

        Parameters
        ----------
        header : header object
           object with fields defining variable header
        process : {True, False} bool, optional
           If True, apply recursive post-processing during loading of
           array.

        Returns
        -------
        arr : array
           array with post-processing applied or not according to
           `process`.
        '''
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        ''' get variables from stream as dictionary

        variable_names   - optional list of variable names to get

        If variable_names is None, then get all variables in file
        '''
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)

        self.mat_stream.seek(0)
        # Here we pass all the parameters in self to the reading objects
        self.initialize_read()
        mdict = self.read_file_header()
        mdict['__globals__'] = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if name in mdict:
                warnings.warn('Duplicate variable name "%s" in stream'
                              ' - replacing previous with new\n'
                              'Consider mio5.varmats_from_mat to split '
                              'file into single variable files' % name,
                              MatReadWarning, stacklevel=2)
            if name == '':
                # can only be a matlab 7 function workspace
                name = '__function_workspace__'
                # We want to keep this raw because mat_dtype processing
                # will break the format (uint8 as mxDOUBLE_CLASS)
                process = False
            else:
                process = True
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            try:
                res = self.read_var_array(hdr, process)
            except MatReadError as err:
                warnings.warn(
                    'Unreadable variable "%s", because "%s"' %
                    (name, err),
                    Warning, stacklevel=2)
                res = "Read error: %s" % err
            self.mat_stream.seek(next_position)
            mdict[name] = res
            if hdr.is_global:
                mdict['__globals__'].append(name)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        if self.simplify_cells:
            return _simplify_cells(mdict)
        else:
            return mdict

    def list_variables(self):
        ''' list variables from stream '''
        self.mat_stream.seek(0)
        # Here we pass all the parameters in self to the reading objects
        self.initialize_read()
        self.read_file_header()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if name == '':
                # can only be a matlab 7 function workspace
                name = '__function_workspace__'

            shape = self._matrix_reader.shape_from_header(hdr)
            if hdr.is_logical:
                info = 'logical'
            else:
                info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))

            self.mat_stream.seek(next_position)
        return vars


def varmats_from_mat(file_obj):
    """ Pull variables out of mat 5 file as a sequence of mat file objects

    This can be useful with a difficult mat file, containing unreadable
    variables. This routine pulls the variables out in raw form and puts them,
    unread, back into a file stream for saving or reading. Another use is the
    pathological case where there is more than one variable of the same name in
    the file; this routine returns the duplicates, whereas the standard reader
    will overwrite duplicates in the returned dictionary.

    The file pointer in `file_obj` will be undefined. File pointers for the
    returned file-like objects are set at 0.

    Parameters
    ----------
    file_obj : file-like
        file object containing mat file

    Returns
    -------
    named_mats : list
        list contains tuples of (name, BytesIO) where BytesIO is a file-like
        object containing mat file contents as for a single variable. The
        BytesIO contains a string with the original header and a single var. If
        ``var_file_obj`` is an individual BytesIO instance, then save as a mat
        file with something like ``open('test.mat',
        'wb').write(var_file_obj.read())``

    Examples
    --------
    >>> import scipy.io

    BytesIO is from the ``io`` module in Python 3, and is ``cStringIO`` for
    Python < 3.

    >>> mat_fileobj = BytesIO()
    >>> scipy.io.savemat(mat_fileobj, {'b': np.arange(10), 'a': 'a string'})
    >>> varmats = varmats_from_mat(mat_fileobj)
    >>> sorted([name for name, str_obj in varmats])
    ['a', 'b']
    """
    rdr = MatFile5Reader(file_obj)
    file_obj.seek(0)
    # Raw read of top-level file header
    hdr_len = MDTYPES[native_code]['dtypes']['file_header'].itemsize
    raw_hdr = file_obj.read(hdr_len)
    # Initialize variable reading
    file_obj.seek(0)
    rdr.initialize_read()
    rdr.read_file_header()
    next_position = file_obj.tell()
    named_mats = []
    while not rdr.end_of_stream():
        start_position = next_position
        hdr, next_position = rdr.read_var_header()
        name = 'None' if hdr.name is None else hdr.name.decode('latin1')
        # Read raw variable string
        file_obj.seek(start_position)
        byte_count = next_position - start_position
        var_str = file_obj.read(byte_count)
        # write to stringio object
        out_obj = BytesIO()
        out_obj.write(raw_hdr)
        out_obj.write(var_str)
        out_obj.seek(0)
        named_mats.append((name, out_obj))
    return named_mats


class EmptyStructMarker:
    """ Class to indicate presence of empty matlab struct on output """


def to_writeable(source):
    ''' Convert input object ``source`` to something we can write

    Parameters
    ----------
    source : object

    Returns
    -------
    arr : None or ndarray or EmptyStructMarker
        If `source` cannot be converted to something we can write to a matfile,
        return None.  If `source` is equivalent to an empty dictionary, return
        ``EmptyStructMarker``.  Otherwise return `source` converted to an
        ndarray with contents for writing to matfile.
    '''
    if isinstance(source, np.ndarray):
        return source
    if source is None:
        return None
    # Objects that implement mappings
    is_mapping = (hasattr(source, 'keys') and hasattr(source, 'values') and
                  hasattr(source, 'items'))
    # Objects that don't implement mappings, but do have dicts
    if isinstance(source, np.generic):
        # NumPy scalars are never mappings (PyPy issue workaround)
        pass
    elif not is_mapping and hasattr(source, '__dict__'):
        source = dict((key, value) for key, value in source.__dict__.items()
                      if not key.startswith('_'))
        is_mapping = True
    if is_mapping:
        dtype = []
        values = []
        for field, value in source.items():
            if (isinstance(field, str) and
                    field[0] not in '_0123456789'):
                dtype.append((str(field), object))
                values.append(value)
        if dtype:
            return np.array([tuple(values)], dtype)
        else:
            return EmptyStructMarker
    # Next try and convert to an array
    narr = np.asanyarray(source)
    if narr.dtype.type in (object, np.object_) and \
       narr.shape == () and narr == source:
        # No interesting conversion possible
        return None
    return narr


# Native byte ordered dtypes for convenience for writers
NDT_FILE_HDR = MDTYPES[native_code]['dtypes']['file_header']
NDT_TAG_FULL = MDTYPES[native_code]['dtypes']['tag_full']
NDT_TAG_SMALL = MDTYPES[native_code]['dtypes']['tag_smalldata']
NDT_ARRAY_FLAGS = MDTYPES[native_code]['dtypes']['array_flags']


class VarWriter5:
    ''' Generic matlab matrix writing class '''
    mat_tag = np.zeros((), NDT_TAG_FULL)
    mat_tag['mdtype'] = miMATRIX

    def __init__(self, file_writer):
        self.file_stream = file_writer.file_stream
        self.unicode_strings = file_writer.unicode_strings
        self.long_field_names = file_writer.long_field_names
        self.oned_as = file_writer.oned_as
        # These are used for top level writes, and unset after
        self._var_name = None
        self._var_is_global = False

    def write_bytes(self, arr):
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        self.file_stream.write(s)

    def write_element(self, arr, mdtype=None):
        ''' write tag and data '''
        if mdtype is None:
            mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
        # Array needs to be in native byte order
        if arr.dtype.byteorder == swapped_code:
            arr = arr.byteswap().newbyteorder()
        byte_count = arr.size*arr.itemsize
        if byte_count <= 4:
            self.write_smalldata_element(arr, mdtype, byte_count)
        else:
            self.write_regular_element(arr, mdtype, byte_count)

    def write_smalldata_element(self, arr, mdtype, byte_count):
        # write tag with embedded data
        tag = np.zeros((), NDT_TAG_SMALL)
        tag['byte_count_mdtype'] = (byte_count << 16) + mdtype
        # if arr.tobytes is < 4, the element will be zero-padded as needed.
        tag['data'] = arr.tobytes(order='F')
        self.write_bytes(tag)

    def write_regular_element(self, arr, mdtype, byte_count):
        # write tag, data
        tag = np.zeros((), NDT_TAG_FULL)
        tag['mdtype'] = mdtype
        tag['byte_count'] = byte_count
        self.write_bytes(tag)
        self.write_bytes(arr)
        # pad to next 64-bit boundary
        bc_mod_8 = byte_count % 8
        if bc_mod_8:
            self.file_stream.write(b'\x00' * (8-bc_mod_8))

    def write_header(self,
                     shape,
                     mclass,
                     is_complex=False,
                     is_logical=False,
                     nzmax=0):
        ''' Write header for given data options
        shape : sequence
           array shape
        mclass      - mat5 matrix class
        is_complex  - True if matrix is complex
        is_logical  - True if matrix is logical
        nzmax        - max non zero elements for sparse arrays

        We get the name and the global flag from the object, and reset
        them to defaults after we've used them
        '''
        # get name and is_global from one-shot object store
        name = self._var_name
        is_global = self._var_is_global
        # initialize the top-level matrix tag, store position
        self._mat_tag_pos = self.file_stream.tell()
        self.write_bytes(self.mat_tag)
        # write array flags (complex, global, logical, class, nzmax)
        af = np.zeros((), NDT_ARRAY_FLAGS)
        af['data_type'] = miUINT32
        af['byte_count'] = 8
        flags = is_complex << 3 | is_global << 2 | is_logical << 1
        af['flags_class'] = mclass | flags << 8
        af['nzmax'] = nzmax
        self.write_bytes(af)
        # shape
        self.write_element(np.array(shape, dtype='i4'))
        # write name
        name = np.asarray(name)
        if name == '':  # empty string zero-terminated
            self.write_smalldata_element(name, miINT8, 0)
        else:
            self.write_element(name, miINT8)
        # reset the one-shot store to defaults
        self._var_name = ''
        self._var_is_global = False

    def update_matrix_tag(self, start_pos):
        curr_pos = self.file_stream.tell()
        self.file_stream.seek(start_pos)
        byte_count = curr_pos - start_pos - 8
        if byte_count >= 2**32:
            raise MatWriteError("Matrix too large to save with Matlab "
                                "5 format")
        self.mat_tag['byte_count'] = byte_count
        self.write_bytes(self.mat_tag)
        self.file_stream.seek(curr_pos)

    def write_top(self, arr, name, is_global):
        """ Write variable at top level of mat file

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        name : str, optional
            name as it will appear in matlab workspace
            default is empty string
        is_global : {False, True}, optional
            whether variable will be global on load into matlab
        """
        # these are set before the top-level header write, and unset at
        # the end of the same write, because they do not apply for lower levels
        self._var_is_global = is_global
        self._var_name = name
        # write the header and data
        self.write(arr)

    def write(self, arr):
        ''' Write `arr` to stream at top and sub levels

        Parameters
        ----------
        arr : array_like
            array-like object to create writer for
        '''
        # store position, so we can update the matrix tag
        mat_tag_pos = self.file_stream.tell()
        # First check if these are sparse
        if scipy.sparse.issparse(arr):
            self.write_sparse(arr)
            self.update_matrix_tag(mat_tag_pos)
            return
        # Try to convert things that aren't arrays
        narr = to_writeable(arr)
        if narr is None:
            raise TypeError('Could not convert %s (type %s) to array'
                            % (arr, type(arr)))
        if isinstance(narr, MatlabObject):
            self.write_object(narr)
        elif isinstance(narr, MatlabFunction):
            raise MatWriteError('Cannot write matlab functions')
        elif narr is EmptyStructMarker:  # empty struct array
            self.write_empty_struct()
        elif narr.dtype.fields:  # struct array
            self.write_struct(narr)
        elif narr.dtype.hasobject:  # cell array
            self.write_cells(narr)
        elif narr.dtype.kind in ('U', 'S'):
            if self.unicode_strings:
                codec = 'UTF8'
            else:
                codec = 'ascii'
            self.write_char(narr, codec)
        else:
            self.write_numeric(narr)
        self.update_matrix_tag(mat_tag_pos)

    def write_numeric(self, arr):
        imagf = arr.dtype.kind == 'c'
        logif = arr.dtype.kind == 'b'
        try:
            mclass = NP_TO_MXTYPES[arr.dtype.str[1:]]
        except KeyError:
            # No matching matlab type, probably complex256 / float128 / float96
            # Cast data to complex128 / float64.
            if imagf:
                arr = arr.astype('c128')
            elif logif:
                arr = arr.astype('i1')  # Should only contain 0/1
            else:
                arr = arr.astype('f8')
            mclass = mxDOUBLE_CLASS
        self.write_header(matdims(arr, self.oned_as),
                          mclass,
                          is_complex=imagf,
                          is_logical=logif)
        if imagf:
            self.write_element(arr.real)
            self.write_element(arr.imag)
        else:
            self.write_element(arr)

    def write_char(self, arr, codec='ascii'):
        ''' Write string array `arr` with given `codec`
        '''
        if arr.size == 0 or np.all(arr == ''):
            # This an empty string array or a string array containing
            # only empty strings. Matlab cannot distinguish between a
            # string array that is empty, and a string array containing
            # only empty strings, because it stores strings as arrays of
            # char. There is no way of having an array of char that is
            # not empty, but contains an empty string. We have to
            # special-case the array-with-empty-strings because even
            # empty strings have zero padding, which would otherwise
            # appear in matlab as a string with a space.
            shape = (0,) * np.max([arr.ndim, 2])
            self.write_header(shape, mxCHAR_CLASS)
            self.write_smalldata_element(arr, miUTF8, 0)
            return
        # non-empty string.
        #
        # Convert to char array
        arr = arr_to_chars(arr)
        # We have to write the shape directly, because we are going
        # recode the characters, and the resulting stream of chars
        # may have a different length
        shape = arr.shape
        self.write_header(shape, mxCHAR_CLASS)
        if arr.dtype.kind == 'U' and arr.size:
            # Make one long string from all the characters. We need to
            # transpose here, because we're flattening the array, before
            # we write the bytes. The bytes have to be written in
            # Fortran order.
            n_chars = np.prod(shape)
            st_arr = np.ndarray(shape=(),
                                dtype=arr_dtype_number(arr, n_chars),
                                buffer=arr.T.copy())  # Fortran order
            # Recode with codec to give byte string
            st = st_arr.item().encode(codec)
            # Reconstruct as 1-D byte array
            arr = np.ndarray(shape=(len(st),),
                             dtype='S1',
                             buffer=st)
        self.write_element(arr, mdtype=miUTF8)

    def write_sparse(self, arr):
        ''' Sparse matrices are 2D
        '''
        A = arr.tocsc()  # convert to sparse CSC format
        A.sort_indices()     # MATLAB expects sorted row indices
        is_complex = (A.dtype.kind == 'c')
        is_logical = (A.dtype.kind == 'b')
        nz = A.nnz
        self.write_header(matdims(arr, self.oned_as),
                          mxSPARSE_CLASS,
                          is_complex=is_complex,
                          is_logical=is_logical,
                          # matlab won't load file with 0 nzmax
                          nzmax=1 if nz == 0 else nz)
        self.write_element(A.indices.astype('i4'))
        self.write_element(A.indptr.astype('i4'))
        self.write_element(A.data.real)
        if is_complex:
            self.write_element(A.data.imag)

    def write_cells(self, arr):
        self.write_header(matdims(arr, self.oned_as),
                          mxCELL_CLASS)
        # loop over data, column major
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            self.write(el)

    def write_empty_struct(self):
        self.write_header((1, 1), mxSTRUCT_CLASS)
        # max field name length set to 1 in an example matlab struct
        self.write_element(np.array(1, dtype=np.int32))
        # Field names element is empty
        self.write_element(np.array([], dtype=np.int8))

    def write_struct(self, arr):
        self.write_header(matdims(arr, self.oned_as),
                          mxSTRUCT_CLASS)
        self._write_items(arr)

    def _write_items(self, arr):
        # write fieldnames
        fieldnames = [f[0] for f in arr.dtype.descr]
        length = max([len(fieldname) for fieldname in fieldnames])+1
        max_length = (self.long_field_names and 64) or 32
        if length > max_length:
            raise ValueError("Field names are restricted to %d characters" %
                             (max_length-1))
        self.write_element(np.array([length], dtype='i4'))
        self.write_element(
            np.array(fieldnames, dtype='S%d' % (length)),
            mdtype=miINT8)
        A = np.atleast_2d(arr).flatten('F')
        for el in A:
            for f in fieldnames:
                self.write(el[f])

    def write_object(self, arr):
        '''Same as writing structs, except different mx class, and extra
        classname element after header
        '''
        self.write_header(matdims(arr, self.oned_as),
                          mxOBJECT_CLASS)
        self.write_element(np.array(arr.classname, dtype='S'),
                           mdtype=miINT8)
        self._write_items(arr)


class MatFile5Writer:
    ''' Class for writing mat5 files '''

    @docfiller
    def __init__(self, file_stream,
                 do_compression=False,
                 unicode_strings=False,
                 global_vars=None,
                 long_field_names=False,
                 oned_as='row'):
        ''' Initialize writer for matlab 5 format files

        Parameters
        ----------
        %(do_compression)s
        %(unicode_strings)s
        global_vars : None or sequence of strings, optional
            Names of variables to be marked as global for matlab
        %(long_fields)s
        %(oned_as)s
        '''
        self.file_stream = file_stream
        self.do_compression = do_compression
        self.unicode_strings = unicode_strings
        if global_vars:
            self.global_vars = global_vars
        else:
            self.global_vars = []
        self.long_field_names = long_field_names
        self.oned_as = oned_as
        self._matrix_writer = None

    def write_file_header(self):
        # write header
        hdr = np.zeros((), NDT_FILE_HDR)
        hdr['description'] = 'MATLAB 5.0 MAT-file Platform: %s, Created on: %s' \
            % (os.name,time.asctime())
        hdr['version'] = 0x0100
        hdr['endian_test'] = np.ndarray(shape=(),
                                      dtype='S2',
                                      buffer=np.uint16(0x4d49))
        self.file_stream.write(hdr.tobytes())

    def put_variables(self, mdict, write_header=None):
        ''' Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` returns name, contents pairs where
           ``name`` which will appear in the matlab workspace in file load, and
           ``contents`` is something writeable to a matlab file, such as a NumPy
           array.
        write_header : {None, True, False}, optional
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        '''
        # write header if requested, or None and start of file
        if write_header is None:
            write_header = self.file_stream.tell() == 0
        if write_header:
            self.write_file_header()
        self._matrix_writer = VarWriter5(self)
        for name, var in mdict.items():
            if name[0] == '_':
                continue
            is_global = name in self.global_vars
            if self.do_compression:
                stream = BytesIO()
                self._matrix_writer.file_stream = stream
                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
                out_str = zlib.compress(stream.getvalue())
                tag = np.empty((), NDT_TAG_FULL)
                tag['mdtype'] = miCOMPRESSED
                tag['byte_count'] = len(out_str)
                self.file_stream.write(tag.tobytes())
                self.file_stream.write(out_str)
            else:  # not compressing
                self._matrix_writer.write_top(var, name.encode('latin1'), is_global)
