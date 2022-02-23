"""
Module for reading and writing matlab (TM) .mat files
"""
# Authors: Travis Oliphant, Matthew Brett

from contextlib import contextmanager

from .miobase import get_matfile_version, docfiller
from .mio4 import MatFile4Reader, MatFile4Writer
from .mio5 import MatFile5Reader, MatFile5Writer

__all__ = ['mat_reader_factory', 'loadmat', 'savemat', 'whosmat']


@contextmanager
def _open_file_context(file_like, appendmat, mode='rb'):
    f, opened = _open_file(file_like, appendmat, mode)
    try:
        yield f
    finally:
        if opened:
            f.close()


def _open_file(file_like, appendmat, mode='rb'):
    """
    Open `file_like` and return as file-like object. First, check if object is
    already file-like; if so, return it as-is. Otherwise, try to pass it
    to open(). If that fails, and `file_like` is a string, and `appendmat` is true,
    append '.mat' and try again.
    """
    reqs = {'read'} if set(mode) & set('r+') else set()
    if set(mode) & set('wax+'):
        reqs.add('write')
    if reqs.issubset(dir(file_like)):
        return file_like, False

    try:
        return open(file_like, mode), True
    except IOError as e:
        # Probably "not found"
        if isinstance(file_like, str):
            if appendmat and not file_like.endswith('.mat'):
                file_like += '.mat'
            return open(file_like, mode), True
        else:
            raise IOError(
                'Reader needs file name or open file-like object'
            ) from e


@docfiller
def mat_reader_factory(file_name, appendmat=True, **kwargs):
    """
    Create reader for matlab .mat format files.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------
    matreader : MatFileReader object
       Initialized instance of MatFileReader class matching the mat file
       type detected in `filename`.
    file_opened : bool
       Whether the file was opened by this routine.

    """
    byte_stream, file_opened = _open_file(file_name, appendmat)
    mjv, mnv = get_matfile_version(byte_stream)
    if mjv == 0:
        return MatFile4Reader(byte_stream, **kwargs), file_opened
    elif mjv == 1:
        return MatFile5Reader(byte_stream, **kwargs), file_opened
    elif mjv == 2:
        raise NotImplementedError('Please use HDF reader for matlab v7.3 files')
    else:
        raise TypeError('Did not recognize version %s' % mjv)


@docfiller
def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
    """
    Load MATLAB file.

    Parameters
    ----------
    file_name : str
       Name of the mat file (do not need .mat extension if
       appendmat==True). Can also pass open file-like object.
    mdict : dict, optional
        Dictionary in which to insert matfile variables.
    appendmat : bool, optional
       True to append the .mat extension to the end of the given
       filename, if not already present.
    byte_order : str or None, optional
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
       struct_as_record=True).
    struct_as_record : bool, optional
       Whether to load MATLAB structs as NumPy record arrays, or as
       old-style NumPy arrays with dtype=object. Setting this flag to
       False replicates the behavior of scipy version 0.7.x (returning
       NumPy object arrays). The default setting is True, because it
       allows easier round-trip load and save of MATLAB files.
    verify_compressed_data_integrity : bool, optional
        Whether the length of compressed sequences in the MATLAB file
        should be checked, to ensure that they are not longer than we expect.
        It is advisable to enable this (the default) because overlong
        compressed sequences in MATLAB files generally indicate that the
        files have experienced some sort of corruption.
    variable_names : None or sequence
        If None (the default) - read all variables in file. Otherwise,
        `variable_names` should be a sequence of strings, giving names of the
        MATLAB variables to read from the file. The reader will skip any
        variable with a name not in this sequence, possibly saving some read
        processing.
    simplify_cells : False, optional
        If True, return a simplified dict structure (which is useful if the mat
        file contains cell arrays). Note that this only affects the structure
        of the result and not its contents (which is identical for both output
        structures). If True, this automatically sets `struct_as_record` to
        False and `squeeze_me` to True, which is required to simplify cells.

    Returns
    -------
    mat_dict : dict
       dictionary with variable names as keys, and loaded matrices as
       values.

    Notes
    -----
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 Python library to read MATLAB 7.3 format mat
    files. Because SciPy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    Examples
    --------
    >>> from os.path import dirname, join as pjoin
    >>> import scipy.io as sio

    Get the filename for an example .mat file from the tests/data directory.

    >>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')
    >>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')

    Load the .mat file contents.

    >>> mat_contents = sio.loadmat(mat_fname)

    The result is a dictionary, one key/value pair for each variable:

    >>> sorted(mat_contents.keys())
    ['__globals__', '__header__', '__version__', 'testdouble']
    >>> mat_contents['testdouble']
    array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,
            3.92699082, 4.71238898, 5.49778714, 6.28318531]])

    By default SciPy reads MATLAB structs as structured NumPy arrays where the
    dtype fields are of type `object` and the names correspond to the MATLAB
    struct field names. This can be disabled by setting the optional argument
    `struct_as_record=False`.

    Get the filename for an example .mat file that contains a MATLAB struct
    called `teststruct` and load the contents.

    >>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')
    >>> matstruct_contents = sio.loadmat(matstruct_fname)
    >>> teststruct = matstruct_contents['teststruct']
    >>> teststruct.dtype
    dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])

    The size of the structured array is the size of the MATLAB struct, not the
    number of elements in any particular field. The shape defaults to 2-D
    unless the optional argument `squeeze_me=True`, in which case all length 1
    dimensions are removed.

    >>> teststruct.size
    1
    >>> teststruct.shape
    (1, 1)

    Get the 'stringfield' of the first element in the MATLAB struct.

    >>> teststruct[0, 0]['stringfield']
    array(['Rats live on no evil star.'],
      dtype='<U26')

    Get the first element of the 'doublefield'.

    >>> teststruct['doublefield'][0, 0]
    array([[ 1.41421356,  2.71828183,  3.14159265]])

    Load the MATLAB struct, squeezing out length 1 dimensions, and get the item
    from the 'complexfield'.

    >>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)
    >>> matstruct_squeezed['teststruct'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].shape
    ()
    >>> matstruct_squeezed['teststruct']['complexfield'].item()
    array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,
        3.14159265+3.14159265j])
    """
    variable_names = kwargs.pop('variable_names', None)
    with _open_file_context(file_name, appendmat) as f:
        MR, _ = mat_reader_factory(f, **kwargs)
        matfile_dict = MR.get_variables(variable_names)

    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict

    return mdict


@docfiller
def savemat(file_name, mdict,
            appendmat=True,
            format='5',
            long_field_names=False,
            do_compression=False,
            oned_as='row'):
    """
    Save a dictionary of names and arrays into a MATLAB-style .mat file.

    This saves the array objects in the given dictionary to a MATLAB-
    style .mat file.

    Parameters
    ----------
    file_name : str or file-like object
        Name of the .mat file (.mat extension not needed if ``appendmat ==
        True``).
        Can also pass open file_like object.
    mdict : dict
        Dictionary from which to save matfile variables.
    appendmat : bool, optional
        True (the default) to append the .mat extension to the end of the
        given filename, if not already present.
    format : {'5', '4'}, string, optional
        '5' (the default) for MATLAB 5 and up (to 7.2),
        '4' for MATLAB 4 .mat files.
    long_field_names : bool, optional
        False (the default) - maximum field name length in a structure is
        31 characters which is the documented maximum length.
        True - maximum field name length in a structure is 63 characters
        which works for MATLAB 7.6+.
    do_compression : bool, optional
        Whether or not to compress matrices on write. Default is False.
    oned_as : {'row', 'column'}, optional
        If 'column', write 1-D NumPy arrays as column vectors.
        If 'row', write 1-D NumPy arrays as row vectors.

    Examples
    --------
    >>> from scipy.io import savemat
    >>> a = np.arange(20)
    >>> mdic = {"a": a, "label": "experiment"}
    >>> mdic
    {'a': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19]),
    'label': 'experiment'}
    >>> savemat("matlab_matrix.mat", mdic)
    """
    with _open_file_context(file_name, appendmat, 'wb') as file_stream:
        if format == '4':
            if long_field_names:
                raise ValueError("Long field names are not available for version 4 files")
            MW = MatFile4Writer(file_stream, oned_as)
        elif format == '5':
            MW = MatFile5Writer(file_stream,
                                do_compression=do_compression,
                                unicode_strings=True,
                                long_field_names=long_field_names,
                                oned_as=oned_as)
        else:
            raise ValueError("Format should be '4' or '5'")
        MW.put_variables(mdict)


@docfiller
def whosmat(file_name, appendmat=True, **kwargs):
    """
    List variables inside a MATLAB file.

    Parameters
    ----------
    %(file_arg)s
    %(append_arg)s
    %(load_args)s
    %(struct_arg)s

    Returns
    -------
    variables : list of tuples
        A list of tuples, where each tuple holds the matrix name (a string),
        its shape (tuple of ints), and its data class (a string).
        Possible data classes are: int8, uint8, int16, uint16, int32, uint32,
        int64, uint64, single, double, cell, struct, object, char, sparse,
        function, opaque, logical, unknown.

    Notes
    -----
    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.

    You will need an HDF5 python library to read matlab 7.3 format mat
    files. Because SciPy does not supply one, we do not implement the
    HDF5 / 7.3 interface here.

    .. versionadded:: 0.12.0

    """
    with _open_file_context(file_name, appendmat) as f:
        ML, file_opened = mat_reader_factory(f, **kwargs)
        variables = ML.list_variables()
    return variables
