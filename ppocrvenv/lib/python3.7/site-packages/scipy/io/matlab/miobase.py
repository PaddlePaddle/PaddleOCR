# Authors: Travis Oliphant, Matthew Brett

"""
Base classes for MATLAB file stream reading.

MATLAB is a registered trademark of the Mathworks inc.
"""
import operator
import functools

import numpy as np
from scipy._lib import doccer

from . import byteordercodes as boc


class MatReadError(Exception):
    pass


class MatWriteError(Exception):
    pass


class MatReadWarning(UserWarning):
    pass


doc_dict = \
    {'file_arg':
         '''file_name : str
   Name of the mat file (do not need .mat extension if
   appendmat==True) Can also pass open file-like object.''',
     'append_arg':
         '''appendmat : bool, optional
   True to append the .mat extension to the end of the given
   filename, if not already present.''',
     'load_args':
         '''byte_order : str or None, optional
   None by default, implying byte order guessed from mat
   file. Otherwise can be one of ('native', '=', 'little', '<',
   'BIG', '>').
mat_dtype : bool, optional
   If True, return arrays in same dtype as would be loaded into
   MATLAB (instead of the dtype with which they are saved).
squeeze_me : bool, optional
   Whether to squeeze unit matrix dimensions or not.
chars_as_strings : bool, optional
   Whether to convert char arrays to string arrays.
matlab_compatible : bool, optional
   Returns matrices as would be loaded by MATLAB (implies
   squeeze_me=False, chars_as_strings=False, mat_dtype=True,
   struct_as_record=True).''',
     'struct_arg':
         '''struct_as_record : bool, optional
   Whether to load MATLAB structs as NumPy record arrays, or as
   old-style NumPy arrays with dtype=object. Setting this flag to
   False replicates the behavior of SciPy version 0.7.x (returning
   numpy object arrays). The default setting is True, because it
   allows easier round-trip load and save of MATLAB files.''',
     'matstream_arg':
         '''mat_stream : file-like
   Object with file API, open for reading.''',
     'long_fields':
         '''long_field_names : bool, optional
   * False - maximum field name length in a structure is 31 characters
     which is the documented maximum length. This is the default.
   * True - maximum field name length in a structure is 63 characters
     which works for MATLAB 7.6''',
     'do_compression':
         '''do_compression : bool, optional
   Whether to compress matrices on write. Default is False.''',
     'oned_as':
         '''oned_as : {'row', 'column'}, optional
   If 'column', write 1-D NumPy arrays as column vectors.
   If 'row', write 1D NumPy arrays as row vectors.''',
     'unicode_strings':
         '''unicode_strings : bool, optional
   If True, write strings as Unicode, else MATLAB usual encoding.'''}

docfiller = doccer.filldoc(doc_dict)

'''

 Note on architecture
======================

There are three sets of parameters relevant for reading files. The
first are *file read parameters* - containing options that are common
for reading the whole file, and therefore every variable within that
file. At the moment these are:

* mat_stream
* dtypes (derived from byte code)
* byte_order
* chars_as_strings
* squeeze_me
* struct_as_record (MATLAB 5 files)
* class_dtypes (derived from order code, MATLAB 5 files)
* codecs (MATLAB 5 files)
* uint16_codec (MATLAB 5 files)

Another set of parameters are those that apply only to the current
variable being read - the *header*:

* header related variables (different for v4 and v5 mat files)
* is_complex
* mclass
* var_stream

With the header, we need ``next_position`` to tell us where the next
variable in the stream is.

Then, for each element in a matrix, there can be *element read
parameters*. An element is, for example, one element in a MATLAB cell
array. At the moment, these are:

* mat_dtype

The file-reading object contains the *file read parameters*. The
*header* is passed around as a data object, or may be read and discarded
in a single function. The *element read parameters* - the mat_dtype in
this instance, is passed into a general post-processing function - see
``mio_utils`` for details.
'''


def convert_dtypes(dtype_template, order_code):
    ''' Convert dtypes in mapping to given order

    Parameters
    ----------
    dtype_template : mapping
       mapping with values returning numpy dtype from ``np.dtype(val)``
    order_code : str
       an order code suitable for using in ``dtype.newbyteorder()``

    Returns
    -------
    dtypes : mapping
       mapping where values have been replaced by
       ``np.dtype(val).newbyteorder(order_code)``

    '''
    dtypes = dtype_template.copy()
    for k in dtypes:
        dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
    return dtypes


def read_dtype(mat_stream, a_dtype):
    """
    Generic get of byte stream data of known type

    Parameters
    ----------
    mat_stream : file_like object
        MATLAB (tm) mat file stream
    a_dtype : dtype
        dtype of array to read. `a_dtype` is assumed to be correct
        endianness.

    Returns
    -------
    arr : ndarray
        Array of dtype `a_dtype` read from stream.

    """
    num_bytes = a_dtype.itemsize
    arr = np.ndarray(shape=(),
                     dtype=a_dtype,
                     buffer=mat_stream.read(num_bytes),
                     order='F')
    return arr


def get_matfile_version(fileobj):
    """
    Return major, minor tuple depending on apparent mat file type

    Where:

     #. 0,x -> version 4 format mat files
     #. 1,x -> version 5 format mat files
     #. 2,x -> version 7.3 format mat files (HDF format)

    Parameters
    ----------
    fileobj : file_like
        object implementing seek() and read()

    Returns
    -------
    major_version : {0, 1, 2}
        major MATLAB File format version
    minor_version : int
        minor MATLAB file format version

    Raises
    ------
    MatReadError
        If the file is empty.
    ValueError
        The matfile version is unknown.

    Notes
    -----
    Has the side effect of setting the file read pointer to 0
    """
    # Mat4 files have a zero somewhere in first 4 bytes
    fileobj.seek(0)
    mopt_bytes = fileobj.read(4)
    if len(mopt_bytes) == 0:
        raise MatReadError("Mat file appears to be empty")
    mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
    if 0 in mopt_ints:
        fileobj.seek(0)
        return (0,0)
    # For 5 format or 7.3 format we need to read an integer in the
    # header. Bytes 124 through 128 contain a version integer and an
    # endian test string
    fileobj.seek(124)
    tst_str = fileobj.read(4)
    fileobj.seek(0)
    maj_ind = int(tst_str[2] == b'I'[0])
    maj_val = int(tst_str[maj_ind])
    min_val = int(tst_str[1 - maj_ind])
    ret = (maj_val, min_val)
    if maj_val in (1, 2):
        return ret
    raise ValueError('Unknown mat file type, version %s, %s' % ret)


def matdims(arr, oned_as='column'):
    """
    Determine equivalent MATLAB dimensions for given array

    Parameters
    ----------
    arr : ndarray
        Input array
    oned_as : {'column', 'row'}, optional
        Whether 1-D arrays are returned as MATLAB row or column matrices.
        Default is 'column'.

    Returns
    -------
    dims : tuple
        Shape tuple, in the form MATLAB expects it.

    Notes
    -----
    We had to decide what shape a 1 dimensional array would be by
    default. ``np.atleast_2d`` thinks it is a row vector. The
    default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.

    Versions of scipy up to and including 0.11 resulted (accidentally)
    in 1-D arrays being read as column vectors. For the moment, we
    maintain the same tradition here.

    Examples
    --------
    >>> matdims(np.array(1)) # NumPy scalar
    (1, 1)
    >>> matdims(np.array([1])) # 1-D array, 1 element
    (1, 1)
    >>> matdims(np.array([1,2])) # 1-D array, 2 elements
    (2, 1)
    >>> matdims(np.array([[2],[3]])) # 2-D array, column vector
    (2, 1)
    >>> matdims(np.array([[2,3]])) # 2-D array, row vector
    (1, 2)
    >>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector
    (1, 1, 2)
    >>> matdims(np.array([])) # empty 1-D array
    (0, 0)
    >>> matdims(np.array([[]])) # empty 2-D array
    (0, 0)
    >>> matdims(np.array([[[]]])) # empty 3-D array
    (0, 0, 0)

    Optional argument flips 1-D shape behavior.

    >>> matdims(np.array([1,2]), 'row') # 1-D array, 2 elements
    (1, 2)

    The argument has to make sense though

    >>> matdims(np.array([1,2]), 'bizarre')
    Traceback (most recent call last):
       ...
    ValueError: 1-D option "bizarre" is strange

    """
    shape = arr.shape
    if shape == ():  # scalar
        return (1, 1)
    if len(shape) == 1:  # 1D
        if shape[0] == 0:
            return (0, 0)
        elif oned_as == 'column':
            return shape + (1,)
        elif oned_as == 'row':
            return (1,) + shape
        else:
            raise ValueError('1-D option "%s" is strange'
                             % oned_as)
    return shape


class MatVarReader:
    ''' Abstract class defining required interface for var readers'''
    def __init__(self, file_reader):
        pass

    def read_header(self):
        ''' Returns header '''
        pass

    def array_from_header(self, header):
        ''' Reads array given header '''
        pass


class MatFileReader:
    """ Base object for reading mat files

    To make this class functional, you will need to override the
    following methods:

    matrix_getter_factory   - gives object to fetch next matrix from stream
    guess_byte_order        - guesses file byte order from file
    """

    @docfiller
    def __init__(self, mat_stream,
                 byte_order=None,
                 mat_dtype=False,
                 squeeze_me=False,
                 chars_as_strings=True,
                 matlab_compatible=False,
                 struct_as_record=True,
                 verify_compressed_data_integrity=True,
                 simplify_cells=False):
        '''
        Initializer for mat file reader

        mat_stream : file-like
            object with file API, open for reading
    %(load_args)s
        '''
        # Initialize stream
        self.mat_stream = mat_stream
        self.dtypes = {}
        if not byte_order:
            byte_order = self.guess_byte_order()
        else:
            byte_order = boc.to_numpy_code(byte_order)
        self.byte_order = byte_order
        self.struct_as_record = struct_as_record
        if matlab_compatible:
            self.set_matlab_compatible()
        else:
            self.squeeze_me = squeeze_me
            self.chars_as_strings = chars_as_strings
            self.mat_dtype = mat_dtype
        self.verify_compressed_data_integrity = verify_compressed_data_integrity
        self.simplify_cells = simplify_cells
        if simplify_cells:
            self.squeeze_me = True
            self.struct_as_record = False

    def set_matlab_compatible(self):
        ''' Sets options to return arrays as MATLAB loads them '''
        self.mat_dtype = True
        self.squeeze_me = False
        self.chars_as_strings = False

    def guess_byte_order(self):
        ''' As we do not know what file type we have, assume native '''
        return boc.native_code

    def end_of_stream(self):
        b = self.mat_stream.read(1)
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos-1)
        return len(b) == 0


def arr_dtype_number(arr, num):
    ''' Return dtype for given number of items per element'''
    return np.dtype(arr.dtype.str[:2] + str(num))


def arr_to_chars(arr):
    ''' Convert string array to char array '''
    dims = list(arr.shape)
    if not dims:
        dims = [1]
    dims.append(int(arr.dtype.str[2:]))
    arr = np.ndarray(shape=dims,
                     dtype=arr_dtype_number(arr, 1),
                     buffer=arr)
    empties = [arr == '']
    if not np.any(empties):
        return arr
    arr = arr.copy()
    arr[tuple(empties)] = ' '
    return arr
