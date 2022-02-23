"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import os
import warnings
try:
    from pathlib import Path
except ImportError:
    Path = None

from .compressor import lz4, LZ4_NOT_INSTALLED_ERROR
from .compressor import _COMPRESSORS, register_compressor, BinaryZlibFile
from .compressor import (ZlibCompressorWrapper, GzipCompressorWrapper,
                         BZ2CompressorWrapper, LZMACompressorWrapper,
                         XZCompressorWrapper, LZ4CompressorWrapper)
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_utils import _ensure_native_byte_order
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
# For compatibility with old versions of joblib, we need ZNDArrayWrapper
# to be visible in the current namespace.
# Explicitly skipping next line from flake8 as it triggers an F401 warning
# which we don't care.
from .numpy_pickle_compat import ZNDArrayWrapper  # noqa
from .backports import make_memmap

# Register supported compressors
register_compressor('zlib', ZlibCompressorWrapper())
register_compressor('gzip', GzipCompressorWrapper())
register_compressor('bz2', BZ2CompressorWrapper())
register_compressor('lzma', LZMACompressorWrapper())
register_compressor('xz', XZCompressorWrapper())
register_compressor('lz4', LZ4CompressorWrapper())


###############################################################################
# Utility objects for persistence.


class NumpyArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    This object is used to hack into the pickle machinery and read numpy
    array data from our custom persistence format.
    More precisely, this object is used for:
    * carrying the information of the persisted array: subclass, shape, order,
    dtype. Those ndarray metadata are used to correctly reconstruct the array
    with low level numpy functions.
    * determining if memmap is allowed on the array.
    * reading the array bytes from a file.
    * reading the array using memorymap from a file.
    * writing the array bytes to a file.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    shape: numpy.ndarray shape
        Determine the shape of the wrapped array.
    order: {'C', 'F'}
        Determine the order of wrapped array data. 'C' is for C order, 'F' is
        for fortran order.
    dtype: numpy.ndarray dtype
        Determine the data type of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
        Default: False.
    """

    def __init__(self, subclass, shape, order, dtype, allow_mmap=False):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.shape = shape
        self.order = order
        self.dtype = dtype
        self.allow_mmap = allow_mmap

    def write_array(self, array, pickler):
        """Write array bytes to pickler file handle.

        This function is an adaptation of the numpy write_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)
        if array.dtype.hasobject:
            # We contain Python objects so we cannot write out the data
            # directly. Instead, we will pickle it out with version 2 of the
            # pickle protocol.
            pickle.dump(array, pickler.file_handle, protocol=2)
        else:
            for chunk in pickler.np.nditer(array,
                                           flags=['external_loop',
                                                  'buffered',
                                                  'zerosize_ok'],
                                           buffersize=buffersize,
                                           order=self.order):
                pickler.file_handle.write(chunk.tobytes('C'))

    def read_array(self, unpickler):
        """Read array from unpickler file handle.

        This function is an adaptation of the numpy read_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        if len(self.shape) == 0:
            count = 1
        else:
            # joblib issue #859: we cast the elements of self.shape to int64 to
            # prevent a potential overflow when computing their product.
            shape_int64 = [unpickler.np.int64(x) for x in self.shape]
            count = unpickler.np.multiply.reduce(shape_int64)
        # Now read the actual data.
        if self.dtype.hasobject:
            # The array contained Python objects. We need to unpickle the data.
            array = pickle.load(unpickler.file_handle)
        else:
            # This is not a real file. We have to read it the
            # memory-intensive way.
            # crc32 module fails on reads greater than 2 ** 32 bytes,
            # breaking large reads from gzip streams. Chunk reads to
            # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
            # of the read. In non-chunked case count < max_read_count, so
            # only one read is performed.
            max_read_count = BUFFER_SIZE // min(BUFFER_SIZE,
                                                self.dtype.itemsize)

            array = unpickler.np.empty(count, dtype=self.dtype)
            for i in range(0, count, max_read_count):
                read_count = min(max_read_count, count - i)
                read_size = int(read_count * self.dtype.itemsize)
                data = _read_bytes(unpickler.file_handle,
                                   read_size, "array data")
                array[i:i + read_count] = \
                    unpickler.np.frombuffer(data, dtype=self.dtype,
                                            count=read_count)
                del data

            if self.order == 'F':
                array.shape = self.shape[::-1]
                array = array.transpose()
            else:
                array.shape = self.shape

        # Detect byte order mis-match and swap as needed.
        return _ensure_native_byte_order(array)

    def read_mmap(self, unpickler):
        """Read an array using numpy memmap."""
        offset = unpickler.file_handle.tell()
        if unpickler.mmap_mode == 'w+':
            unpickler.mmap_mode = 'r+'

        marray = make_memmap(unpickler.filename,
                             dtype=self.dtype,
                             shape=self.shape,
                             order=self.order,
                             mode=unpickler.mmap_mode,
                             offset=offset)
        # update the offset so that it corresponds to the end of the read array
        unpickler.file_handle.seek(offset + marray.nbytes)

        return marray

    def read(self, unpickler):
        """Read the array corresponding to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler

        Returns
        -------
        array: numpy.ndarray

        """
        # When requested, only use memmap mode if allowed.
        if unpickler.mmap_mode is not None and self.allow_mmap:
            array = self.read_mmap(unpickler)
        else:
            array = self.read_array(unpickler)

        # Manage array subclass case
        if (hasattr(array, '__array_prepare__') and
            self.subclass not in (unpickler.np.ndarray,
                                  unpickler.np.memmap)):
            # We need to reconstruct another subclass
            new_array = unpickler.np.core.multiarray._reconstruct(
                self.subclass, (0,), 'b')
            return new_array.__array_prepare__(array)
        else:
            return array

###############################################################################
# Pickler classes


class NumpyPickler(Pickler):
    """A pickler to persist big data efficiently.

    The main features of this object are:
    * persistence of numpy arrays in a single file.
    * optional compression with a special care on avoiding memory copies.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    protocol: int, optional
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL.
    """

    dispatch = Pickler.dispatch.copy()

    def __init__(self, fp, protocol=None):
        self.file_handle = fp
        self.buffered = isinstance(self.file_handle, BinaryZlibFile)

        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        if protocol is None:
            protocol = pickle.DEFAULT_PROTOCOL

        Pickler.__init__(self, self.file_handle, protocol=protocol)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array."""
        order = 'F' if (array.flags.f_contiguous and
                        not array.flags.c_contiguous) else 'C'
        allow_mmap = not self.buffered and not array.dtype.hasobject
        wrapper = NumpyArrayWrapper(type(array),
                                    array.shape, order, array.dtype,
                                    allow_mmap=allow_mmap)

        return wrapper

    def save(self, obj):
        """Subclass the Pickler `save` method.

        This is a total abuse of the Pickler class in order to use the numpy
        persistence function `save` instead of the default pickle
        implementation. The numpy array is replaced by a custom wrapper in the
        pickle persistence stack and the serialized array is written right
        after in the file. Warning: the file produced does not follow the
        pickle format. As such it can not be read with `pickle.load`.
        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                                 self.np.matrix,
                                                 self.np.memmap):
            if type(obj) is self.np.memmap:
                # Pickling doesn't work with memmapped arrays
                obj = self.np.asanyarray(obj)

            # The array wrapper is pickled instead of the real array.
            wrapper = self._create_array_wrapper(obj)
            Pickler.save(self, wrapper)

            # A framer was introduced with pickle protocol 4 and we want to
            # ensure the wrapper object is written before the numpy array
            # buffer in the pickle file.
            # See https://www.python.org/dev/peps/pep-3154/#framing to get
            # more information on the framer behavior.
            if self.proto >= 4:
                self.framer.commit_frame(force=True)

            # And then array bytes are written right after the wrapper.
            wrapper.write_array(obj, self)
            return

        return Pickler.save(self, obj)


class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.

    Attributes
    ----------
    mmap_mode: str
        The memorymap mode to use for reading numpy arrays.
    file_handle: file_like
        File object to unpickle from.
    filename: str
        Name of the file to unpickle from. It should correspond to file_handle.
        This parameter is required when using mmap_mode.
    np: module
        Reference to numpy module if numpy is installed else None.

    """

    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        # The next line is for backward compatibility with pickle generated
        # with joblib versions less than 0.10.
        self._dirname = os.path.dirname(filename)

        self.mmap_mode = mmap_mode
        self.file_handle = file_handle
        # filename is required for numpy mmap mode.
        self.filename = filename
        self.compat_mode = False
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def load_build(self):
        """Called to set the state of a newly created object.

        We capture it to replace our place-holder objects, NDArrayWrapper or
        NumpyArrayWrapper, by the array we are interested in. We
        replace them directly in the stack of pickler.
        NDArrayWrapper is used for backward compatibility with joblib <= 0.9.
        """
        Unpickler.load_build(self)

        # For backward compatibility, we support NDArrayWrapper objects.
        if isinstance(self.stack[-1], (NDArrayWrapper, NumpyArrayWrapper)):
            if self.np is None:
                raise ImportError("Trying to unpickle an ndarray, "
                                  "but numpy didn't import correctly")
            array_wrapper = self.stack.pop()
            # If any NDArrayWrapper is found, we switch to compatibility mode,
            # this will be used to raise a DeprecationWarning to the user at
            # the end of the unpickling.
            if isinstance(array_wrapper, NDArrayWrapper):
                self.compat_mode = True
            self.stack.append(array_wrapper.read(self))

    # Be careful to register our new method.
    dispatch[pickle.BUILD[0]] = load_build


###############################################################################
# Utility functions

def dump(value, filename, compress=0, protocol=None, cache_size=None):
    """Persist an arbitrary Python object into one file.

    Read more in the :ref:`User Guide <persistence>`.

    Parameters
    -----------
    value: any Python object
        The object to store to disk.
    filename: str, pathlib.Path, or file object.
        The file object or path of the file in which it is to be stored.
        The compression method corresponding to one of the supported filename
        extensions ('.z', '.gz', '.bz2', '.xz' or '.lzma') will be used
        automatically.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
        If compress is a 2-tuple, the first element must correspond to a string
        between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
        'xz'), the second element must be an integer from 0 to 9, corresponding
        to the compression level.
    protocol: int, optional
        Pickle protocol, see pickle.dump documentation for more details.
    cache_size: positive int, optional
        This option is deprecated in 0.10 and has no effect.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    joblib.load : corresponding loader

    Notes
    -----
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take extra extra memory during
    dump and load.

    """

    if Path is not None and isinstance(filename, Path):
        filename = str(filename)

    is_filename = isinstance(filename, str)
    is_fileobj = hasattr(filename, "write")

    compress_method = 'zlib'  # zlib is the default compression method.
    if compress is True:
        # By default, if compress is enabled, we want the default compress
        # level of the compressor.
        compress_level = None
    elif isinstance(compress, tuple):
        # a 2-tuple was set in compress
        if len(compress) != 2:
            raise ValueError(
                'Compress argument tuple should contain exactly 2 elements: '
                '(compress method, compress level), you passed {}'
                .format(compress))
        compress_method, compress_level = compress
    elif isinstance(compress, str):
        compress_method = compress
        compress_level = None  # Use default compress level
        compress = (compress_method, compress_level)
    else:
        compress_level = compress

    if compress_method == 'lz4' and lz4 is None:
        raise ValueError(LZ4_NOT_INSTALLED_ERROR)

    if (compress_level is not None and
            compress_level is not False and
            compress_level not in range(10)):
        # Raising an error if a non valid compress level is given.
        raise ValueError(
            'Non valid compress level given: "{}". Possible values are '
            '{}.'.format(compress_level, list(range(10))))

    if compress_method not in _COMPRESSORS:
        # Raising an error if an unsupported compression method is given.
        raise ValueError(
            'Non valid compression method given: "{}". Possible values are '
            '{}.'.format(compress_method, _COMPRESSORS))

    if not is_filename and not is_fileobj:
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
            'Second argument should be a filename or a file-like object, '
            '%s (type %s) was given.'
            % (filename, type(filename))
        )

    if is_filename and not isinstance(compress, tuple):
        # In case no explicit compression was requested using both compression
        # method and level in a tuple and the filename has an explicit
        # extension, we select the corresponding compressor.

        # unset the variable to be sure no compression level is set afterwards.
        compress_method = None
        for name, compressor in _COMPRESSORS.items():
            if filename.endswith(compressor.extension):
                compress_method = name

        if compress_method in _COMPRESSORS and compress_level == 0:
            # we choose the default compress_level in case it was not given
            # as an argument (using compress).
            compress_level = None

    if cache_size is not None:
        # Cache size is deprecated starting from version 0.10
        warnings.warn("Please do not set 'cache_size' in joblib.dump, "
                      "this parameter has no effect and will be removed. "
                      "You used 'cache_size={}'".format(cache_size),
                      DeprecationWarning, stacklevel=2)

    if compress_level != 0:
        with _write_fileobject(filename, compress=(compress_method,
                                                   compress_level)) as f:
            NumpyPickler(f, protocol=protocol).dump(value)
    elif is_filename:
        with open(filename, 'wb') as f:
            NumpyPickler(f, protocol=protocol).dump(value)
    else:
        NumpyPickler(filename, protocol=protocol).dump(value)

    # If the target container is a file object, nothing is returned.
    if is_fileobj:
        return

    # For compatibility, the list of created filenames (e.g with one element
    # after 0.10.0) is returned by default.
    return [filename]


def _unpickle(fobj, filename="", mmap_mode=None):
    """Internal unpickling function."""
    # We are careful to open the file handle early and keep it open to
    # avoid race-conditions on renames.
    # That said, if data is stored in companion files, which can be
    # the case with the old persistence format, moving the directory
    # will create a race when joblib tries to access the companion
    # files.
    unpickler = NumpyUnpickler(filename, fobj, mmap_mode=mmap_mode)
    obj = None
    try:
        obj = unpickler.load()
        if unpickler.compat_mode:
            warnings.warn("The file '%s' has been generated with a "
                          "joblib version less than 0.10. "
                          "Please regenerate this pickle file."
                          % filename,
                          DeprecationWarning, stacklevel=3)
    except UnicodeDecodeError as exc:
        # More user-friendly error message
        new_exc = ValueError(
            'You may be trying to read with '
            'python 3 a joblib pickle generated with python 2. '
            'This feature is not supported by joblib.')
        new_exc.__cause__ = exc
        raise new_exc
    return obj


def load_temporary_memmap(filename, mmap_mode, unlink_on_gc_collect):
    from ._memmapping_reducer import JOBLIB_MMAPS, add_maybe_unlink_finalizer
    obj = load(filename, mmap_mode)
    JOBLIB_MMAPS.add(obj.filename)
    if unlink_on_gc_collect:
        add_maybe_unlink_finalizer(obj)
    return obj


def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Read more in the :ref:`User Guide <persistence>`.

    WARNING: joblib.load relies on the pickle module and can therefore
    execute arbitrary Python code. It should therefore never be used
    to load files from untrusted sources.

    Parameters
    -----------
    filename: str, pathlib.Path, or file object.
        The file object or path of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
        case the reconstructed object might no longer match exactly
        the originally pickled object.

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmapped.
    """
    if Path is not None and isinstance(filename, Path):
        filename = str(filename)

    if hasattr(filename, "read"):
        fobj = filename
        filename = getattr(fobj, 'name', '')
        with _read_fileobject(fobj, filename, mmap_mode) as fobj:
            obj = _unpickle(fobj)
    else:
        with open(filename, 'rb') as f:
            with _read_fileobject(f, filename, mmap_mode) as fobj:
                if isinstance(fobj, str):
                    # if the returned file object is a string, this means we
                    # try to load a pickle file generated with an version of
                    # Joblib so we load it with joblib compatibility function.
                    return load_compatibility(fobj)

                obj = _unpickle(fobj, filename, mmap_mode)
    return obj
