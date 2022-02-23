"""Common IO api utilities"""
from __future__ import annotations

import bz2
import codecs
from collections import abc
import dataclasses
import gzip
from io import (
    BufferedIOBase,
    BytesIO,
    RawIOBase,
    StringIO,
    TextIOWrapper,
)
import mmap
import os
import tempfile
from typing import (
    IO,
    Any,
    AnyStr,
    Mapping,
    cast,
)
from urllib.parse import (
    urljoin,
    urlparse as parse_url,
    uses_netloc,
    uses_params,
    uses_relative,
)
import warnings
import zipfile

from pandas._typing import (
    Buffer,
    CompressionDict,
    CompressionOptions,
    FileOrBuffer,
    FilePathOrBuffer,
    StorageOptions,
)
from pandas.compat import (
    get_lzma_file,
    import_lzma,
)
from pandas.compat._optional import import_optional_dependency

from pandas.core.dtypes.common import is_file_like

lzma = import_lzma()


_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")


@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.

    Note (copy&past from io/parsers):
    filepath_or_buffer can be Union[FilePathOrBuffer, s3fs.S3File, gcsfs.GCSFile]
    though mypy handling of conditional imports is difficult.
    See https://github.com/python/mypy/issues/1297
    """

    filepath_or_buffer: FileOrBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = False


@dataclasses.dataclass
class IOHandles:
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    handle: Buffer
    compression: CompressionDict
    created_handles: list[Buffer] = dataclasses.field(default_factory=list)
    is_wrapped: bool = False
    is_mmap: bool = False

    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        try:
            for handle in self.created_handles:
                handle.close()
        except (OSError, ValueError):
            pass
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self) -> IOHandles:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def is_url(url) -> bool:
    """
    Check to see if a URL has a valid protocol.

    Parameters
    ----------
    url : str or unicode

    Returns
    -------
    isurl : bool
        If `url` has a valid protocol return True otherwise False.
    """
    if not isinstance(url, str):
        return False
    return parse_url(url).scheme in _VALID_URLS


def _expand_user(filepath_or_buffer: FileOrBuffer[AnyStr]) -> FileOrBuffer[AnyStr]:
    """
    Return the argument with an initial component of ~ or ~user
    replaced by that user's home directory.

    Parameters
    ----------
    filepath_or_buffer : object to be converted if possible

    Returns
    -------
    expanded_filepath_or_buffer : an expanded filepath or the
                                  input if not expandable
    """
    if isinstance(filepath_or_buffer, str):
        return os.path.expanduser(filepath_or_buffer)
    return filepath_or_buffer


def validate_header_arg(header) -> None:
    if isinstance(header, bool):
        raise TypeError(
            "Passing a bool to header is invalid. Use header=None for no header or "
            "header=int or list-like of ints to specify "
            "the row(s) making up the column names"
        )


def stringify_path(
    filepath_or_buffer: FilePathOrBuffer[AnyStr],
    convert_file_like: bool = False,
) -> FileOrBuffer[AnyStr]:
    """
    Attempt to convert a path-like object to a string.

    Parameters
    ----------
    filepath_or_buffer : object to be converted

    Returns
    -------
    str_filepath_or_buffer : maybe a string version of the object

    Notes
    -----
    Objects supporting the fspath protocol (python 3.6+) are coerced
    according to its __fspath__ method.

    Any other object is passed through unchanged, which includes bytes,
    strings, buffers, or anything else that's not even path-like.
    """
    if not convert_file_like and is_file_like(filepath_or_buffer):
        # GH 38125: some fsspec objects implement os.PathLike but have already opened a
        # file. This prevents opening the file a second time. infer_compression calls
        # this function with convert_file_like=True to infer the compression.
        return cast(FileOrBuffer[AnyStr], filepath_or_buffer)

    if isinstance(filepath_or_buffer, os.PathLike):
        filepath_or_buffer = filepath_or_buffer.__fspath__()
    return _expand_user(filepath_or_buffer)


def urlopen(*args, **kwargs):
    """
    Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
    the stdlib.
    """
    import urllib.request

    return urllib.request.urlopen(*args, **kwargs)


def is_fsspec_url(url: FilePathOrBuffer) -> bool:
    """
    Returns true if the given URL looks like
    something fsspec can handle
    """
    return (
        isinstance(url, str)
        and "://" in url
        and not url.startswith(("http://", "https://"))
    )


def _get_filepath_or_buffer(
    filepath_or_buffer: FilePathOrBuffer,
    encoding: str = "utf-8",
    compression: CompressionOptions = None,
    mode: str = "r",
    storage_options: StorageOptions = None,
) -> IOArgs:
    """
    If the filepath_or_buffer is a url, translate and return the buffer.
    Otherwise passthrough.

    Parameters
    ----------
    filepath_or_buffer : a url, filepath (str, py.path.local or pathlib.Path),
                         or buffer
    compression : {{'gzip', 'bz2', 'zip', 'xz', None}}, optional
    encoding : the encoding to use to decode bytes, default is 'utf-8'
    mode : str, optional

    storage_options : dict, optional
        Extra options that make sense for a particular storage connection, e.g.
        host, port, username, password, etc., if using a URL that will
        be parsed by ``fsspec``, e.g., starting "s3://", "gcs://". An error
        will be raised if providing this argument with a local path or
        a file-like buffer. See the fsspec and backend storage implementation
        docs for the set of allowed keys and values

        .. versionadded:: 1.2.0

    ..versionchange:: 1.2.0

      Returns the dataclass IOArgs.
    """
    filepath_or_buffer = stringify_path(filepath_or_buffer)

    # handle compression dict
    compression_method, compression = get_compression_method(compression)
    compression_method = infer_compression(filepath_or_buffer, compression_method)

    # GH21227 internal compression is not used for non-binary handles.
    if compression_method and hasattr(filepath_or_buffer, "write") and "b" not in mode:
        warnings.warn(
            "compression has no effect when passing a non-binary object as input.",
            RuntimeWarning,
            stacklevel=2,
        )
        compression_method = None

    compression = dict(compression, method=compression_method)

    # uniform encoding names
    if encoding is not None:
        encoding = encoding.replace("_", "-").lower()

    # bz2 and xz do not write the byte order mark for utf-16 and utf-32
    # print a warning when writing such files
    if (
        "w" in mode
        and compression_method in ["bz2", "xz"]
        and encoding in ["utf-16", "utf-32"]
    ):
        warnings.warn(
            f"{compression} will not write the byte order mark for {encoding}",
            UnicodeWarning,
        )

    # Use binary mode when converting path-like objects to file-like objects (fsspec)
    # except when text mode is explicitly requested. The original mode is returned if
    # fsspec is not used.
    fsspec_mode = mode
    if "t" not in fsspec_mode and "b" not in fsspec_mode:
        fsspec_mode += "b"

    if isinstance(filepath_or_buffer, str) and is_url(filepath_or_buffer):
        # TODO: fsspec can also handle HTTP via requests, but leaving this
        # unchanged. using fsspec appears to break the ability to infer if the
        # server responded with gzipped data
        storage_options = storage_options or {}

        # waiting until now for importing to match intended lazy logic of
        # urlopen function defined elsewhere in this module
        import urllib.request

        # assuming storage_options is to be interpreted as headers
        req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
        with urlopen(req_info) as req:
            content_encoding = req.headers.get("Content-Encoding", None)
            if content_encoding == "gzip":
                # Override compression based on Content-Encoding header
                compression = {"method": "gzip"}
            reader = BytesIO(req.read())
        return IOArgs(
            filepath_or_buffer=reader,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )

    if is_fsspec_url(filepath_or_buffer):
        assert isinstance(
            filepath_or_buffer, str
        )  # just to appease mypy for this branch
        # two special-case s3-like protocols; these have special meaning in Hadoop,
        # but are equivalent to just "s3" from fsspec's point of view
        # cc #11071
        if filepath_or_buffer.startswith("s3a://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3a://", "s3://")
        if filepath_or_buffer.startswith("s3n://"):
            filepath_or_buffer = filepath_or_buffer.replace("s3n://", "s3://")
        fsspec = import_optional_dependency("fsspec")

        # If botocore is installed we fallback to reading with anon=True
        # to allow reads from public buckets
        err_types_to_retry_with_anon: list[Any] = []
        try:
            import_optional_dependency("botocore")
            from botocore.exceptions import (
                ClientError,
                NoCredentialsError,
            )

            err_types_to_retry_with_anon = [
                ClientError,
                NoCredentialsError,
                PermissionError,
            ]
        except ImportError:
            pass

        try:
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()
        # GH 34626 Reads from Public Buckets without Credentials needs anon=True
        except tuple(err_types_to_retry_with_anon):
            if storage_options is None:
                storage_options = {"anon": True}
            else:
                # don't mutate user input.
                storage_options = dict(storage_options)
                storage_options["anon"] = True
            file_obj = fsspec.open(
                filepath_or_buffer, mode=fsspec_mode, **(storage_options or {})
            ).open()

        return IOArgs(
            filepath_or_buffer=file_obj,
            encoding=encoding,
            compression=compression,
            should_close=True,
            mode=fsspec_mode,
        )
    elif storage_options:
        raise ValueError(
            "storage_options passed with file object or non-fsspec file path"
        )

    if isinstance(filepath_or_buffer, (str, bytes, mmap.mmap)):
        return IOArgs(
            filepath_or_buffer=_expand_user(filepath_or_buffer),
            encoding=encoding,
            compression=compression,
            should_close=False,
            mode=mode,
        )

    if not is_file_like(filepath_or_buffer):
        msg = f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
        raise ValueError(msg)

    return IOArgs(
        filepath_or_buffer=filepath_or_buffer,
        encoding=encoding,
        compression=compression,
        should_close=False,
        mode=mode,
    )


def file_path_to_url(path: str) -> str:
    """
    converts an absolute native path to a FILE URL.

    Parameters
    ----------
    path : a path in native format

    Returns
    -------
    a valid FILE URL
    """
    # lazify expensive import (~30ms)
    from urllib.request import pathname2url

    return urljoin("file:", pathname2url(path))


_compression_to_extension = {"gzip": ".gz", "bz2": ".bz2", "zip": ".zip", "xz": ".xz"}


def get_compression_method(
    compression: CompressionOptions,
) -> tuple[str | None, CompressionDict]:
    """
    Simplifies a compression argument to a compression method string and
    a mapping containing additional arguments.

    Parameters
    ----------
    compression : str or mapping
        If string, specifies the compression method. If mapping, value at key
        'method' specifies compression method.

    Returns
    -------
    tuple of ({compression method}, Optional[str]
              {compression arguments}, Dict[str, Any])

    Raises
    ------
    ValueError on mapping missing 'method' key
    """
    compression_method: str | None
    if isinstance(compression, Mapping):
        compression_args = dict(compression)
        try:
            compression_method = compression_args.pop("method")
        except KeyError as err:
            raise ValueError("If mapping, compression must have key 'method'") from err
    else:
        compression_args = {}
        compression_method = compression
    return compression_method, compression_args


def infer_compression(
    filepath_or_buffer: FilePathOrBuffer, compression: str | None
) -> str | None:
    """
    Get the compression method for filepath_or_buffer. If compression='infer',
    the inferred compression method is returned. Otherwise, the input
    compression method is returned unchanged, unless it's invalid, in which
    case an error is raised.

    Parameters
    ----------
    filepath_or_buffer : str or file handle
        File path or object.
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
        If 'infer' and `filepath_or_buffer` is path-like, then detect
        compression from the following extensions: '.gz', '.bz2', '.zip',
        or '.xz' (otherwise no compression).

    Returns
    -------
    string or None

    Raises
    ------
    ValueError on invalid compression specified.
    """
    if compression is None:
        return None

    # Infer compression
    if compression == "infer":
        # Convert all path types (e.g. pathlib.Path) to strings
        filepath_or_buffer = stringify_path(filepath_or_buffer, convert_file_like=True)
        if not isinstance(filepath_or_buffer, str):
            # Cannot infer compression of a buffer, assume no compression
            return None

        # Infer compression from the filename/URL extension
        for compression, extension in _compression_to_extension.items():
            if filepath_or_buffer.lower().endswith(extension):
                return compression
        return None

    # Compression has been specified. Check that it's valid
    if compression in _compression_to_extension:
        return compression

    # https://github.com/python/mypy/issues/5492
    # Unsupported operand types for + ("List[Optional[str]]" and "List[str]")
    valid = ["infer", None] + sorted(
        _compression_to_extension
    )  # type: ignore[operator]
    msg = (
        f"Unrecognized compression type: {compression}\n"
        f"Valid compression types are {valid}"
    )
    raise ValueError(msg)


def get_handle(
    path_or_buf: FilePathOrBuffer,
    mode: str,
    encoding: str | None = None,
    compression: CompressionOptions = None,
    memory_map: bool = False,
    is_text: bool = True,
    errors: str | None = None,
    storage_options: StorageOptions = None,
) -> IOHandles:
    """
    Get file handle for given path/buffer and mode.

    Parameters
    ----------
    path_or_buf : str or file handle
        File path or object.
    mode : str
        Mode to open path_or_buf with.
    encoding : str or None
        Encoding to use.
    compression : str or dict, default None
        If string, specifies compression mode. If dict, value at key 'method'
        specifies compression mode. Compression mode must be one of {'infer',
        'gzip', 'bz2', 'zip', 'xz', None}. If compression mode is 'infer'
        and `filepath_or_buffer` is path-like, then detect compression from
        the following extensions: '.gz', '.bz2', '.zip', or '.xz' (otherwise
        no compression). If dict and compression mode is one of
        {'zip', 'gzip', 'bz2'}, or inferred as one of the above,
        other entries passed as additional compression options.

        .. versionchanged:: 1.0.0

           May now be a dict with key 'method' as compression mode
           and other keys as compression options if compression
           mode is 'zip'.

        .. versionchanged:: 1.1.0

           Passing compression options as keys in dict is now
           supported for compression modes 'gzip' and 'bz2' as well as 'zip'.

    memory_map : bool, default False
        See parsers._parser_params for more information.
    is_text : bool, default True
        Whether the type of the content passed to the file/buffer is string or
        bytes. This is not the same as `"b" not in mode`. If a string content is
        passed to a binary file/buffer, a wrapper is inserted.
    errors : str, default 'strict'
        Specifies how encoding and decoding errors are to be handled.
        See the errors argument for :func:`open` for a full list
        of options.
    storage_options: StorageOptions = None
        Passed to _get_filepath_or_buffer

    .. versionchanged:: 1.2.0

    Returns the dataclass IOHandles
    """
    # Windows does not default to utf-8. Set to utf-8 for a consistent behavior
    encoding = encoding or "utf-8"

    # read_csv does not know whether the buffer is opened in binary/text mode
    if _is_binary_mode(path_or_buf, mode) and "b" not in mode:
        mode += "b"

    # valdiate errors
    if isinstance(errors, str):
        errors = errors.lower()
    if errors not in (
        None,
        "strict",
        "ignore",
        "replace",
        "xmlcharrefreplace",
        "backslashreplace",
        "namereplace",
        "surrogateescape",
        "surrogatepass",
    ):
        raise ValueError(
            f"Invalid value for `encoding_errors` ({errors}). Please see "
            + "https://docs.python.org/3/library/codecs.html#error-handlers "
            + "for valid values."
        )

    # open URLs
    ioargs = _get_filepath_or_buffer(
        path_or_buf,
        encoding=encoding,
        compression=compression,
        mode=mode,
        storage_options=storage_options,
    )

    handle = ioargs.filepath_or_buffer
    handles: list[Buffer]

    # memory mapping needs to be the first step
    handle, memory_map, handles = _maybe_memory_map(
        handle,
        memory_map,
        ioargs.encoding,
        ioargs.mode,
        errors,
        ioargs.compression["method"] not in _compression_to_extension,
    )

    is_path = isinstance(handle, str)
    compression_args = dict(ioargs.compression)
    compression = compression_args.pop("method")

    if compression:
        # compression libraries do not like an explicit text-mode
        ioargs.mode = ioargs.mode.replace("t", "")

        # GZ Compression
        if compression == "gzip":
            if is_path:
                assert isinstance(handle, str)
                handle = gzip.GzipFile(
                    filename=handle,
                    mode=ioargs.mode,
                    **compression_args,
                )
            else:
                handle = gzip.GzipFile(
                    # error: Argument "fileobj" to "GzipFile" has incompatible type
                    # "Union[str, Union[IO[Any], RawIOBase, BufferedIOBase, TextIOBase,
                    # TextIOWrapper, mmap]]"; expected "Optional[IO[bytes]]"
                    fileobj=handle,  # type: ignore[arg-type]
                    mode=ioargs.mode,
                    **compression_args,
                )

        # BZ Compression
        elif compression == "bz2":
            handle = bz2.BZ2File(
                # Argument 1 to "BZ2File" has incompatible type "Union[str,
                # Union[IO[Any], RawIOBase, BufferedIOBase, TextIOBase, TextIOWrapper,
                # mmap]]"; expected "Union[Union[str, bytes, _PathLike[str],
                # _PathLike[bytes]], IO[bytes]]"
                handle,  # type: ignore[arg-type]
                mode=ioargs.mode,
                **compression_args,
            )

        # ZIP Compression
        elif compression == "zip":
            handle = _BytesZipFile(handle, ioargs.mode, **compression_args)
            if handle.mode == "r":
                handles.append(handle)
                zip_names = handle.namelist()
                if len(zip_names) == 1:
                    handle = handle.open(zip_names.pop())
                elif len(zip_names) == 0:
                    raise ValueError(f"Zero files found in ZIP file {path_or_buf}")
                else:
                    raise ValueError(
                        "Multiple files found in ZIP file. "
                        f"Only one file per ZIP: {zip_names}"
                    )

        # XZ Compression
        elif compression == "xz":
            handle = get_lzma_file(lzma)(handle, ioargs.mode)

        # Unrecognized Compression
        else:
            msg = f"Unrecognized compression type: {compression}"
            raise ValueError(msg)

        assert not isinstance(handle, str)
        handles.append(handle)

    elif isinstance(handle, str):
        # Check whether the filename is to be opened in binary mode.
        # Binary mode does not support 'encoding' and 'newline'.
        if ioargs.encoding and "b" not in ioargs.mode:
            # Encoding
            handle = open(
                handle,
                ioargs.mode,
                encoding=ioargs.encoding,
                errors=errors,
                newline="",
            )
        else:
            # Binary mode
            handle = open(handle, ioargs.mode)
        handles.append(handle)

    # Convert BytesIO or file objects passed with an encoding
    is_wrapped = False
    if is_text and (compression or _is_binary_mode(handle, ioargs.mode)):
        handle = TextIOWrapper(
            # error: Argument 1 to "TextIOWrapper" has incompatible type
            # "Union[IO[bytes], IO[Any], RawIOBase, BufferedIOBase, TextIOBase, mmap]";
            # expected "IO[bytes]"
            handle,  # type: ignore[arg-type]
            encoding=ioargs.encoding,
            errors=errors,
            newline="",
        )
        handles.append(handle)
        # only marked as wrapped when the caller provided a handle
        is_wrapped = not (
            isinstance(ioargs.filepath_or_buffer, str) or ioargs.should_close
        )

    handles.reverse()  # close the most recently added buffer first
    if ioargs.should_close:
        assert not isinstance(ioargs.filepath_or_buffer, str)
        handles.append(ioargs.filepath_or_buffer)

    assert not isinstance(handle, str)
    return IOHandles(
        handle=handle,
        created_handles=handles,
        is_wrapped=is_wrapped,
        is_mmap=memory_map,
        compression=ioargs.compression,
    )


# error: Definition of "__exit__" in base class "ZipFile" is incompatible with
# definition in base class "BytesIO"  [misc]
# error: Definition of "__enter__" in base class "ZipFile" is incompatible with
# definition in base class "BytesIO"  [misc]
# error: Definition of "__enter__" in base class "ZipFile" is incompatible with
# definition in base class "BinaryIO"  [misc]
# error: Definition of "__enter__" in base class "ZipFile" is incompatible with
# definition in base class "IO"  [misc]
# error: Definition of "read" in base class "ZipFile" is incompatible with
# definition in base class "BytesIO"  [misc]
# error: Definition of "read" in base class "ZipFile" is incompatible with
# definition in base class "IO"  [misc]
class _BytesZipFile(zipfile.ZipFile, BytesIO):  # type: ignore[misc]
    """
    Wrapper for standard library class ZipFile and allow the returned file-like
    handle to accept byte strings via `write` method.

    BytesIO provides attributes of file-like object and ZipFile.writestr writes
    bytes strings into a member of the archive.
    """

    # GH 17778
    def __init__(
        self,
        file: FilePathOrBuffer,
        mode: str,
        archive_name: str | None = None,
        **kwargs,
    ):
        mode = mode.replace("b", "")
        self.archive_name = archive_name
        self.multiple_write_buffer: StringIO | BytesIO | None = None

        kwargs_zip: dict[str, Any] = {"compression": zipfile.ZIP_DEFLATED}
        kwargs_zip.update(kwargs)

        # error: Argument 1 to "__init__" of "ZipFile" has incompatible type
        # "Union[_PathLike[str], Union[str, Union[IO[Any], RawIOBase, BufferedIOBase,
        # TextIOBase, TextIOWrapper, mmap]]]"; expected "Union[Union[str,
        # _PathLike[str]], IO[bytes]]"
        super().__init__(file, mode, **kwargs_zip)  # type: ignore[arg-type]

    def write(self, data):
        # buffer multiple write calls, write on flush
        if self.multiple_write_buffer is None:
            self.multiple_write_buffer = (
                BytesIO() if isinstance(data, bytes) else StringIO()
            )
        self.multiple_write_buffer.write(data)

    def flush(self) -> None:
        # write to actual handle and close write buffer
        if self.multiple_write_buffer is None or self.multiple_write_buffer.closed:
            return

        # ZipFile needs a non-empty string
        archive_name = self.archive_name or self.filename or "zip"
        with self.multiple_write_buffer:
            super().writestr(archive_name, self.multiple_write_buffer.getvalue())

    def close(self):
        self.flush()
        super().close()

    @property
    def closed(self):
        return self.fp is None


class _MMapWrapper(abc.Iterator):
    """
    Wrapper for the Python's mmap class so that it can be properly read in
    by Python's csv.reader class.

    Parameters
    ----------
    f : file object
        File object to be mapped onto memory. Must support the 'fileno'
        method or have an equivalent attribute

    """

    def __init__(
        self,
        f: IO,
        encoding: str = "utf-8",
        errors: str = "strict",
        decode: bool = True,
    ):
        self.encoding = encoding
        self.errors = errors
        self.decoder = codecs.getincrementaldecoder(encoding)(errors=errors)
        self.decode = decode

        self.attributes = {}
        for attribute in ("seekable", "readable", "writeable"):
            if not hasattr(f, attribute):
                continue
            self.attributes[attribute] = getattr(f, attribute)()
        self.mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __getattr__(self, name: str):
        if name in self.attributes:
            return lambda: self.attributes[name]
        return getattr(self.mmap, name)

    def __iter__(self) -> _MMapWrapper:
        return self

    def read(self, size: int = -1) -> str | bytes:
        # CSV c-engine uses read instead of iterating
        content: bytes = self.mmap.read(size)
        if self.decode:
            # memory mapping is applied before compression. Encoding should
            # be applied to the de-compressed data.
            final = size == -1 or len(content) < size
            return self.decoder.decode(content, final=final)
        return content

    def __next__(self) -> str:
        newbytes = self.mmap.readline()

        # readline returns bytes, not str, but Python's CSV reader
        # expects str, so convert the output to str before continuing
        newline = self.decoder.decode(newbytes)

        # mmap doesn't raise if reading past the allocated
        # data but instead returns an empty string, so raise
        # if that is returned
        if newline == "":
            raise StopIteration

        # IncrementalDecoder seems to push newline to the next line
        return newline.lstrip("\n")


def _maybe_memory_map(
    handle: FileOrBuffer,
    memory_map: bool,
    encoding: str,
    mode: str,
    errors: str | None,
    decode: bool,
) -> tuple[FileOrBuffer, bool, list[Buffer]]:
    """Try to memory map file/buffer."""
    handles: list[Buffer] = []
    memory_map &= hasattr(handle, "fileno") or isinstance(handle, str)
    if not memory_map:
        return handle, memory_map, handles

    # need to open the file first
    if isinstance(handle, str):
        if encoding and "b" not in mode:
            # Encoding
            handle = open(handle, mode, encoding=encoding, errors=errors, newline="")
        else:
            # Binary mode
            handle = open(handle, mode)
        handles.append(handle)

    try:
        # error: Argument 1 to "_MMapWrapper" has incompatible type "Union[IO[Any],
        # RawIOBase, BufferedIOBase, TextIOBase, mmap]"; expected "IO[Any]"
        wrapped = cast(
            mmap.mmap,
            _MMapWrapper(handle, encoding, errors, decode),  # type: ignore[arg-type]
        )
        handle.close()
        handles.remove(handle)
        handles.append(wrapped)
        handle = wrapped
    except Exception:
        # we catch any errors that may have occurred
        # because that is consistent with the lower-level
        # functionality of the C engine (pd.read_csv), so
        # leave the file handler as is then
        memory_map = False

    return handle, memory_map, handles


def file_exists(filepath_or_buffer: FilePathOrBuffer) -> bool:
    """Test whether file exists."""
    exists = False
    filepath_or_buffer = stringify_path(filepath_or_buffer)
    if not isinstance(filepath_or_buffer, str):
        return exists
    try:
        exists = os.path.exists(filepath_or_buffer)
        # gh-5874: if the filepath is too long will raise here
    except (TypeError, ValueError):
        pass
    return exists


def _is_binary_mode(handle: FilePathOrBuffer, mode: str) -> bool:
    """Whether the handle is opened in binary mode"""
    # specified by user
    if "t" in mode or "b" in mode:
        return "b" in mode

    # exceptions
    text_classes = (
        # classes that expect string but have 'b' in mode
        codecs.StreamWriter,
        codecs.StreamReader,
        codecs.StreamReaderWriter,
        # cannot be wrapped in TextIOWrapper GH43439
        tempfile.SpooledTemporaryFile,
    )
    if issubclass(type(handle), text_classes):
        return False

    # classes that expect bytes
    binary_classes = (BufferedIOBase, RawIOBase)
    return isinstance(handle, binary_classes) or "b" in getattr(handle, "mode", mode)
