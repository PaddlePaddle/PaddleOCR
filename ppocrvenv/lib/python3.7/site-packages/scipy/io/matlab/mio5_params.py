''' Constants and classes for matlab 5 read and write

See also mio5_utils.pyx where these same constants arise as c enums.

If you make changes in this file, don't forget to change mio5_utils.pyx
'''
import numpy as np

from .miobase import convert_dtypes

miINT8 = 1
miUINT8 = 2
miINT16 = 3
miUINT16 = 4
miINT32 = 5
miUINT32 = 6
miSINGLE = 7
miDOUBLE = 9
miINT64 = 12
miUINT64 = 13
miMATRIX = 14
miCOMPRESSED = 15
miUTF8 = 16
miUTF16 = 17
miUTF32 = 18

mxCELL_CLASS = 1
mxSTRUCT_CLASS = 2
# The March 2008 edition of "Matlab 7 MAT-File Format" says that
# mxOBJECT_CLASS = 3, whereas matrix.h says that mxLOGICAL = 3.
# Matlab 2008a appears to save logicals as type 9, so we assume that
# the document is correct. See type 18, below.
mxOBJECT_CLASS = 3
mxCHAR_CLASS = 4
mxSPARSE_CLASS = 5
mxDOUBLE_CLASS = 6
mxSINGLE_CLASS = 7
mxINT8_CLASS = 8
mxUINT8_CLASS = 9
mxINT16_CLASS = 10
mxUINT16_CLASS = 11
mxINT32_CLASS = 12
mxUINT32_CLASS = 13
# The following are not in the March 2008 edition of "Matlab 7
# MAT-File Format," but were guessed from matrix.h.
mxINT64_CLASS = 14
mxUINT64_CLASS = 15
mxFUNCTION_CLASS = 16
# Not doing anything with these at the moment.
mxOPAQUE_CLASS = 17  # This appears to be a function workspace
# Thread 'saving/loading symbol table of annymous functions', octave-maintainers, April-May 2007
# https://lists.gnu.org/archive/html/octave-maintainers/2007-04/msg00031.html
# https://lists.gnu.org/archive/html/octave-maintainers/2007-05/msg00032.html
# (Was/Deprecated: https://www-old.cae.wisc.edu/pipermail/octave-maintainers/2007-May/002824.html)
mxOBJECT_CLASS_FROM_MATRIX_H = 18

mdtypes_template = {
    miINT8: 'i1',
    miUINT8: 'u1',
    miINT16: 'i2',
    miUINT16: 'u2',
    miINT32: 'i4',
    miUINT32: 'u4',
    miSINGLE: 'f4',
    miDOUBLE: 'f8',
    miINT64: 'i8',
    miUINT64: 'u8',
    miUTF8: 'u1',
    miUTF16: 'u2',
    miUTF32: 'u4',
    'file_header': [('description', 'S116'),
                    ('subsystem_offset', 'i8'),
                    ('version', 'u2'),
                    ('endian_test', 'S2')],
    'tag_full': [('mdtype', 'u4'), ('byte_count', 'u4')],
    'tag_smalldata':[('byte_count_mdtype', 'u4'), ('data', 'S4')],
    'array_flags': [('data_type', 'u4'),
                    ('byte_count', 'u4'),
                    ('flags_class','u4'),
                    ('nzmax', 'u4')],
    'U1': 'U1',
    }

mclass_dtypes_template = {
    mxINT8_CLASS: 'i1',
    mxUINT8_CLASS: 'u1',
    mxINT16_CLASS: 'i2',
    mxUINT16_CLASS: 'u2',
    mxINT32_CLASS: 'i4',
    mxUINT32_CLASS: 'u4',
    mxINT64_CLASS: 'i8',
    mxUINT64_CLASS: 'u8',
    mxSINGLE_CLASS: 'f4',
    mxDOUBLE_CLASS: 'f8',
    }

mclass_info = {
    mxINT8_CLASS: 'int8',
    mxUINT8_CLASS: 'uint8',
    mxINT16_CLASS: 'int16',
    mxUINT16_CLASS: 'uint16',
    mxINT32_CLASS: 'int32',
    mxUINT32_CLASS: 'uint32',
    mxINT64_CLASS: 'int64',
    mxUINT64_CLASS: 'uint64',
    mxSINGLE_CLASS: 'single',
    mxDOUBLE_CLASS: 'double',
    mxCELL_CLASS: 'cell',
    mxSTRUCT_CLASS: 'struct',
    mxOBJECT_CLASS: 'object',
    mxCHAR_CLASS: 'char',
    mxSPARSE_CLASS: 'sparse',
    mxFUNCTION_CLASS: 'function',
    mxOPAQUE_CLASS: 'opaque',
    }

NP_TO_MTYPES = {
    'f8': miDOUBLE,
    'c32': miDOUBLE,
    'c24': miDOUBLE,
    'c16': miDOUBLE,
    'f4': miSINGLE,
    'c8': miSINGLE,
    'i8': miINT64,
    'i4': miINT32,
    'i2': miINT16,
    'i1': miINT8,
    'u8': miUINT64,
    'u4': miUINT32,
    'u2': miUINT16,
    'u1': miUINT8,
    'S1': miUINT8,
    'U1': miUTF16,
    'b1': miUINT8,  # not standard but seems MATLAB uses this (gh-4022)
    }


NP_TO_MXTYPES = {
    'f8': mxDOUBLE_CLASS,
    'c32': mxDOUBLE_CLASS,
    'c24': mxDOUBLE_CLASS,
    'c16': mxDOUBLE_CLASS,
    'f4': mxSINGLE_CLASS,
    'c8': mxSINGLE_CLASS,
    'i8': mxINT64_CLASS,
    'i4': mxINT32_CLASS,
    'i2': mxINT16_CLASS,
    'i1': mxINT8_CLASS,
    'u8': mxUINT64_CLASS,
    'u4': mxUINT32_CLASS,
    'u2': mxUINT16_CLASS,
    'u1': mxUINT8_CLASS,
    'S1': mxUINT8_CLASS,
    'b1': mxUINT8_CLASS,  # not standard but seems MATLAB uses this
    }

''' Before release v7.1 (release 14) matlab (TM) used the system
default character encoding scheme padded out to 16-bits. Release 14
and later use Unicode. When saving character data, R14 checks if it
can be encoded in 7-bit ascii, and saves in that format if so.'''

codecs_template = {
    miUTF8: {'codec': 'utf_8', 'width': 1},
    miUTF16: {'codec': 'utf_16', 'width': 2},
    miUTF32: {'codec': 'utf_32','width': 4},
    }


def _convert_codecs(template, byte_order):
    ''' Convert codec template mapping to byte order

    Set codecs not on this system to None

    Parameters
    ----------
    template : mapping
       key, value are respectively codec name, and root name for codec
       (without byte order suffix)
    byte_order : {'<', '>'}
       code for little or big endian

    Returns
    -------
    codecs : dict
       key, value are name, codec (as in .encode(codec))
    '''
    codecs = {}
    postfix = byte_order == '<' and '_le' or '_be'
    for k, v in template.items():
        codec = v['codec']
        try:
            " ".encode(codec)
        except LookupError:
            codecs[k] = None
            continue
        if v['width'] > 1:
            codec += postfix
        codecs[k] = codec
    return codecs.copy()


MDTYPES = {}
for _bytecode in '<>':
    _def = {'dtypes': convert_dtypes(mdtypes_template, _bytecode),
            'classes': convert_dtypes(mclass_dtypes_template, _bytecode),
            'codecs': _convert_codecs(codecs_template, _bytecode)}
    MDTYPES[_bytecode] = _def


class mat_struct:
    ''' Placeholder for holding read data from structs

    We use instances of this class when the user passes False as a value to the
    ``struct_as_record`` parameter of the :func:`scipy.io.matlab.loadmat`
    function.
    '''
    pass


class MatlabObject(np.ndarray):
    ''' ndarray Subclass to contain matlab object '''
    def __new__(cls, input_array, classname=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.classname = classname
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self,obj):
        # reset the attribute from passed original object
        self.classname = getattr(obj, 'classname', None)
        # We do not need to return anything


class MatlabFunction(np.ndarray):
    ''' Subclass to signal this is a matlab function '''
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


class MatlabOpaque(np.ndarray):
    ''' Subclass to signal this is a matlab opaque matrix '''
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj


OPAQUE_DTYPE = np.dtype(
    [('s0', 'O'), ('s1', 'O'), ('s2', 'O'), ('arr', 'O')])
