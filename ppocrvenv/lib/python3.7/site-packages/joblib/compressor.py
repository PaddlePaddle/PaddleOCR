"""Classes and functions for managing compressors."""

import io
import zlib
from distutils.version import LooseVersion

try:
    from threading import RLock
except ImportError:
    from dummy_threading import RLock

try:
    import bz2
except ImportError:
    bz2 = None

try:
    import lz4
    from lz4.frame import LZ4FrameFile
except ImportError:
    lz4 = None

try:
    import lzma
except ImportError:
    lzma = None


LZ4_NOT_INSTALLED_ERROR = ('LZ4 is not installed. Install it with pip: '
                           'https://python-lz4.readthedocs.io/')

# Registered compressors
_COMPRESSORS = {}

# Magic numbers of supported compression file formats.
_ZFILE_PREFIX = b'ZF'  # used with pickle files created before 0.9.3.
_ZLIB_PREFIX = b'\x78'
_GZIP_PREFIX = b'\x1f\x8b'
_BZ2_PREFIX = b'BZ'
_XZ_PREFIX = b'\xfd\x37\x7a\x58\x5a'
_LZMA_PREFIX = b'\x5d\x00'
_LZ4_PREFIX = b'\x04\x22\x4D\x18'


def register_compressor(compressor_name, compressor,
                        force=False):
    """Register a new compressor.

    Parameters
    -----------
    compressor_name: str.
        The name of the compressor.
    compressor: CompressorWrapper
        An instance of a 'CompressorWrapper'.
    """
    global _COMPRESSORS
    if not isinstance(compressor_name, str):
        raise ValueError("Compressor name should be a string, "
                         "'{}' given.".format(compressor_name))

    if not isinstance(compressor, CompressorWrapper):
        raise ValueError("Compressor should implement the CompressorWrapper "
                         "interface, '{}' given.".format(compressor))

    if (compressor.fileobj_factory is not None and
            (not hasattr(compressor.fileobj_factory, 'read') or
             not hasattr(compressor.fileobj_factory, 'write') or
             not hasattr(compressor.fileobj_factory, 'seek') or
             not hasattr(compressor.fileobj_factory, 'tell'))):
        raise ValueError("Compressor 'fileobj_factory' attribute should "
                         "implement the file object interface, '{}' given."
                         .format(compressor.fileobj_factory))

    if compressor_name in _COMPRESSORS and not force:
        raise ValueError("Compressor '{}' already registered."
                         .format(compressor_name))

    _COMPRESSORS[compressor_name] = compressor


class CompressorWrapper():
    """A wrapper around a compressor file object.

    Attributes
    ----------
    obj: a file-like object
        The object must implement the buffer interface and will be used
        internally to compress/decompress the data.
    prefix: bytestring
        A bytestring corresponding to the magic number that identifies the
        file format associated to the compressor.
    extention: str
        The file extension used to automatically select this compressor during
        a dump to a file.
    """

    def __init__(self, obj, prefix=b'', extension=''):
        self.fileobj_factory = obj
        self.prefix = prefix
        self.extension = extension

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        if compresslevel is None:
            return self.fileobj_factory(fileobj, 'wb')
        else:
            return self.fileobj_factory(fileobj, 'wb',
                                        compresslevel=compresslevel)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        return self.fileobj_factory(fileobj, 'rb')


class BZ2CompressorWrapper(CompressorWrapper):

    prefix = _BZ2_PREFIX
    extension = '.bz2'

    def __init__(self):
        if bz2 is not None:
            self.fileobj_factory = bz2.BZ2File
        else:
            self.fileobj_factory = None

    def _check_versions(self):
        if bz2 is None:
            raise ValueError('bz2 module is not compiled on your python '
                             'standard library.')

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        self._check_versions()
        if compresslevel is None:
            return self.fileobj_factory(fileobj, 'wb')
        else:
            return self.fileobj_factory(fileobj, 'wb',
                                        compresslevel=compresslevel)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        self._check_versions()
        fileobj = self.fileobj_factory(fileobj, 'rb')
        return fileobj


class LZMACompressorWrapper(CompressorWrapper):

    prefix = _LZMA_PREFIX
    extension = '.lzma'
    _lzma_format_name = 'FORMAT_ALONE'

    def __init__(self):
        if lzma is not None:
            self.fileobj_factory = lzma.LZMAFile
            self._lzma_format = getattr(lzma, self._lzma_format_name)
        else:
            self.fileobj_factory = None

    def _check_versions(self):
        if lzma is None:
            raise ValueError('lzma module is not compiled on your python '
                             'standard library.')

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        if compresslevel is None:
            return self.fileobj_factory(fileobj, 'wb',
                                        format=self._lzma_format)
        else:
            return self.fileobj_factory(fileobj, 'wb',
                                        format=self._lzma_format,
                                        preset=compresslevel)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        return lzma.LZMAFile(fileobj, 'rb')


class XZCompressorWrapper(LZMACompressorWrapper):

    prefix = _XZ_PREFIX
    extension = '.xz'
    _lzma_format_name = 'FORMAT_XZ'


class LZ4CompressorWrapper(CompressorWrapper):

    prefix = _LZ4_PREFIX
    extension = '.lz4'

    def __init__(self):
        if lz4 is not None:
            self.fileobj_factory = LZ4FrameFile
        else:
            self.fileobj_factory = None

    def _check_versions(self):
        if lz4 is None:
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)
        lz4_version = lz4.__version__
        if lz4_version.startswith("v"):
            lz4_version = lz4_version[1:]
        if LooseVersion(lz4_version) < LooseVersion('0.19'):
            raise ValueError(LZ4_NOT_INSTALLED_ERROR)

    def compressor_file(self, fileobj, compresslevel=None):
        """Returns an instance of a compressor file object."""
        self._check_versions()
        if compresslevel is None:
            return self.fileobj_factory(fileobj, 'wb')
        else:
            return self.fileobj_factory(fileobj, 'wb',
                                        compression_level=compresslevel)

    def decompressor_file(self, fileobj):
        """Returns an instance of a decompressor file object."""
        self._check_versions()
        return self.fileobj_factory(fileobj, 'rb')


###############################################################################
#  base file compression/decompression object definition
_MODE_CLOSED = 0
_MODE_READ = 1
_MODE_READ_EOF = 2
_MODE_WRITE = 3
_BUFFER_SIZE = 8192


class BinaryZlibFile(io.BufferedIOBase):
    """A file object providing transparent zlib (de)compression.

    TODO python2_drop: is it still needed since we dropped Python 2 support A
    BinaryZlibFile can act as a wrapper for an existing file object, or refer
    directly to a named file on disk.

    Note that BinaryZlibFile provides only a *binary* file interface: data read
    is returned as bytes, and data to be written should be given as bytes.

    This object is an adaptation of the BZ2File object and is compatible with
    versions of python >= 2.7.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """

    wbits = zlib.MAX_WBITS

    def __init__(self, filename, mode="rb", compresslevel=3):
        # This lock must be recursive, so that BufferedIOBase's
        # readline(), readlines() and writelines() don't deadlock.
        self._lock = RLock()
        self._fp = None
        self._closefp = False
        self._mode = _MODE_CLOSED
        self._pos = 0
        self._size = -1
        self.compresslevel = compresslevel

        if not isinstance(compresslevel, int) or not (1 <= compresslevel <= 9):
            raise ValueError("'compresslevel' must be an integer "
                             "between 1 and 9. You provided 'compresslevel={}'"
                             .format(compresslevel))

        if mode == "rb":
            self._mode = _MODE_READ
            self._decompressor = zlib.decompressobj(self.wbits)
            self._buffer = b""
            self._buffer_offset = 0
        elif mode == "wb":
            self._mode = _MODE_WRITE
            self._compressor = zlib.compressobj(self.compresslevel,
                                                zlib.DEFLATED, self.wbits,
                                                zlib.DEF_MEM_LEVEL, 0)
        else:
            raise ValueError("Invalid mode: %r" % (mode,))

        if isinstance(filename, str):
            self._fp = io.open(filename, mode)
            self._closefp = True
        elif hasattr(filename, "read") or hasattr(filename, "write"):
            self._fp = filename
        else:
            raise TypeError("filename must be a str or bytes object, "
                            "or a file")

    def close(self):
        """Flush and close the file.

        May be called more than once without error. Once the file is
        closed, any other operation on it will raise a ValueError.
        """
        with self._lock:
            if self._mode == _MODE_CLOSED:
                return
            try:
                if self._mode in (_MODE_READ, _MODE_READ_EOF):
                    self._decompressor = None
                elif self._mode == _MODE_WRITE:
                    self._fp.write(self._compressor.flush())
                    self._compressor = None
            finally:
                try:
                    if self._closefp:
                        self._fp.close()
                finally:
                    self._fp = None
                    self._closefp = False
                    self._mode = _MODE_CLOSED
                    self._buffer = b""
                    self._buffer_offset = 0

    @property
    def closed(self):
        """True if this file is closed."""
        return self._mode == _MODE_CLOSED

    def fileno(self):
        """Return the file descriptor for the underlying file."""
        self._check_not_closed()
        return self._fp.fileno()

    def seekable(self):
        """Return whether the file supports seeking."""
        return self.readable() and self._fp.seekable()

    def readable(self):
        """Return whether the file was opened for reading."""
        self._check_not_closed()
        return self._mode in (_MODE_READ, _MODE_READ_EOF)

    def writable(self):
        """Return whether the file was opened for writing."""
        self._check_not_closed()
        return self._mode == _MODE_WRITE

    # Mode-checking helper functions.

    def _check_not_closed(self):
        if self.closed:
            fname = getattr(self._fp, 'name', None)
            msg = "I/O operation on closed file"
            if fname is not None:
                msg += " {}".format(fname)
            msg += "."
            raise ValueError(msg)

    def _check_can_read(self):
        if self._mode not in (_MODE_READ, _MODE_READ_EOF):
            self._check_not_closed()
            raise io.UnsupportedOperation("File not open for reading")

    def _check_can_write(self):
        if self._mode != _MODE_WRITE:
            self._check_not_closed()
            raise io.UnsupportedOperation("File not open for writing")

    def _check_can_seek(self):
        if self._mode not in (_MODE_READ, _MODE_READ_EOF):
            self._check_not_closed()
            raise io.UnsupportedOperation("Seeking is only supported "
                                          "on files open for reading")
        if not self._fp.seekable():
            raise io.UnsupportedOperation("The underlying file object "
                                          "does not support seeking")

    # Fill the readahead buffer if it is empty. Returns False on EOF.
    def _fill_buffer(self):
        if self._mode == _MODE_READ_EOF:
            return False
        # Depending on the input data, our call to the decompressor may not
        # return any data. In this case, try again after reading another block.
        while self._buffer_offset == len(self._buffer):
            try:
                rawblock = (self._decompressor.unused_data or
                            self._fp.read(_BUFFER_SIZE))
                if not rawblock:
                    raise EOFError
            except EOFError:
                # End-of-stream marker and end of file. We're good.
                self._mode = _MODE_READ_EOF
                self._size = self._pos
                return False
            else:
                self._buffer = self._decompressor.decompress(rawblock)
            self._buffer_offset = 0
        return True

    # Read data until EOF.
    # If return_data is false, consume the data without returning it.
    def _read_all(self, return_data=True):
        # The loop assumes that _buffer_offset is 0. Ensure that this is true.
        self._buffer = self._buffer[self._buffer_offset:]
        self._buffer_offset = 0

        blocks = []
        while self._fill_buffer():
            if return_data:
                blocks.append(self._buffer)
            self._pos += len(self._buffer)
            self._buffer = b""
        if return_data:
            return b"".join(blocks)

    # Read a block of up to n bytes.
    # If return_data is false, consume the data without returning it.
    def _read_block(self, n_bytes, return_data=True):
        # If we have enough data buffered, return immediately.
        end = self._buffer_offset + n_bytes
        if end <= len(self._buffer):
            data = self._buffer[self._buffer_offset: end]
            self._buffer_offset = end
            self._pos += len(data)
            return data if return_data else None

        # The loop assumes that _buffer_offset is 0. Ensure that this is true.
        self._buffer = self._buffer[self._buffer_offset:]
        self._buffer_offset = 0

        blocks = []
        while n_bytes > 0 and self._fill_buffer():
            if n_bytes < len(self._buffer):
                data = self._buffer[:n_bytes]
                self._buffer_offset = n_bytes
            else:
                data = self._buffer
                self._buffer = b""
            if return_data:
                blocks.append(data)
            self._pos += len(data)
            n_bytes -= len(data)
        if return_data:
            return b"".join(blocks)

    def read(self, size=-1):
        """Read up to size uncompressed bytes from the file.

        If size is negative or omitted, read until EOF is reached.
        Returns b'' if the file is already at EOF.
        """
        with self._lock:
            self._check_can_read()
            if size == 0:
                return b""
            elif size < 0:
                return self._read_all()
            else:
                return self._read_block(size)

    def readinto(self, b):
        """Read up to len(b) bytes into b.

        Returns the number of bytes read (0 for EOF).
        """
        with self._lock:
            return io.BufferedIOBase.readinto(self, b)

    def write(self, data):
        """Write a byte string to the file.

        Returns the number of uncompressed bytes written, which is
        always len(data). Note that due to buffering, the file on disk
        may not reflect the data written until close() is called.
        """
        with self._lock:
            self._check_can_write()
            # Convert data type if called by io.BufferedWriter.
            if isinstance(data, memoryview):
                data = data.tobytes()

            compressed = self._compressor.compress(data)
            self._fp.write(compressed)
            self._pos += len(data)
            return len(data)

    # Rewind the file to the beginning of the data stream.
    def _rewind(self):
        self._fp.seek(0, 0)
        self._mode = _MODE_READ
        self._pos = 0
        self._decompressor = zlib.decompressobj(self.wbits)
        self._buffer = b""
        self._buffer_offset = 0

    def seek(self, offset, whence=0):
        """Change the file position.

        The new position is specified by offset, relative to the
        position indicated by whence. Values for whence are:

            0: start of stream (default); offset must not be negative
            1: current stream position
            2: end of stream; offset must not be positive

        Returns the new file position.

        Note that seeking is emulated, so depending on the parameters,
        this operation may be extremely slow.
        """
        with self._lock:
            self._check_can_seek()

            # Recalculate offset as an absolute file position.
            if whence == 0:
                pass
            elif whence == 1:
                offset = self._pos + offset
            elif whence == 2:
                # Seeking relative to EOF - we need to know the file's size.
                if self._size < 0:
                    self._read_all(return_data=False)
                offset = self._size + offset
            else:
                raise ValueError("Invalid value for whence: %s" % (whence,))

            # Make it so that offset is the number of bytes to skip forward.
            if offset < self._pos:
                self._rewind()
            else:
                offset -= self._pos

            # Read and discard data until we reach the desired position.
            self._read_block(offset, return_data=False)

            return self._pos

    def tell(self):
        """Return the current file position."""
        with self._lock:
            self._check_not_closed()
            return self._pos


class ZlibCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryZlibFile,
                                   prefix=_ZLIB_PREFIX, extension='.z')


class BinaryGzipFile(BinaryZlibFile):
    """A file object providing transparent gzip (de)compression.

    If filename is a str or bytes object, it gives the name
    of the file to be opened. Otherwise, it should be a file object,
    which will be used to read or write the compressed data.

    mode can be 'rb' for reading (default) or 'wb' for (over)writing

    If mode is 'wb', compresslevel can be a number between 1
    and 9 specifying the level of compression: 1 produces the least
    compression, and 9 produces the most compression. 3 is the default.
    """

    wbits = 31  # zlib compressor/decompressor wbits value for gzip format.


class GzipCompressorWrapper(CompressorWrapper):

    def __init__(self):
        CompressorWrapper.__init__(self, obj=BinaryGzipFile,
                                   prefix=_GZIP_PREFIX, extension='.gz')
