import io
import re
import typing as t
from functools import partial
from functools import update_wrapper
from itertools import chain

from ._internal import _make_encode_wrapper
from ._internal import _to_bytes
from ._internal import _to_str
from .sansio import utils as _sansio_utils
from .sansio.utils import host_is_trusted  # noqa: F401 # Imported as part of API
from .urls import _URLTuple
from .urls import uri_to_iri
from .urls import url_join
from .urls import url_parse
from .urls import url_quote

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


def responder(f: t.Callable[..., "WSGIApplication"]) -> "WSGIApplication":
    """Marks a function as responder.  Decorate a function with it and it
    will automatically call the return value as WSGI application.

    Example::

        @responder
        def application(environ, start_response):
            return Response('Hello World!')
    """
    return update_wrapper(lambda *a: f(*a)(*a[-2:]), f)


def get_current_url(
    environ: "WSGIEnvironment",
    root_only: bool = False,
    strip_querystring: bool = False,
    host_only: bool = False,
    trusted_hosts: t.Optional[t.Iterable[str]] = None,
) -> str:
    """Recreate the URL for a request from the parts in a WSGI
    environment.

    The URL is an IRI, not a URI, so it may contain Unicode characters.
    Use :func:`~werkzeug.urls.iri_to_uri` to convert it to ASCII.

    :param environ: The WSGI environment to get the URL parts from.
    :param root_only: Only build the root path, don't include the
        remaining path or query string.
    :param strip_querystring: Don't include the query string.
    :param host_only: Only build the scheme and host.
    :param trusted_hosts: A list of trusted host names to validate the
        host against.
    """
    parts = {
        "scheme": environ["wsgi.url_scheme"],
        "host": get_host(environ, trusted_hosts),
    }

    if not host_only:
        parts["root_path"] = environ.get("SCRIPT_NAME", "")

        if not root_only:
            parts["path"] = environ.get("PATH_INFO", "")

            if not strip_querystring:
                parts["query_string"] = environ.get("QUERY_STRING", "").encode("latin1")

    return _sansio_utils.get_current_url(**parts)


def _get_server(
    environ: "WSGIEnvironment",
) -> t.Optional[t.Tuple[str, t.Optional[int]]]:
    name = environ.get("SERVER_NAME")

    if name is None:
        return None

    try:
        port: t.Optional[int] = int(environ.get("SERVER_PORT", None))
    except (TypeError, ValueError):
        # unix socket
        port = None

    return name, port


def get_host(
    environ: "WSGIEnvironment", trusted_hosts: t.Optional[t.Iterable[str]] = None
) -> str:
    """Return the host for the given WSGI environment.

    The ``Host`` header is preferred, then ``SERVER_NAME`` if it's not
    set. The returned host will only contain the port if it is different
    than the standard port for the protocol.

    Optionally, verify that the host is trusted using
    :func:`host_is_trusted` and raise a
    :exc:`~werkzeug.exceptions.SecurityError` if it is not.

    :param environ: A WSGI environment dict.
    :param trusted_hosts: A list of trusted host names.

    :return: Host, with port if necessary.
    :raise ~werkzeug.exceptions.SecurityError: If the host is not
        trusted.
    """
    return _sansio_utils.get_host(
        environ["wsgi.url_scheme"],
        environ.get("HTTP_HOST"),
        _get_server(environ),
        trusted_hosts,
    )


def get_content_length(environ: "WSGIEnvironment") -> t.Optional[int]:
    """Returns the content length from the WSGI environment as
    integer. If it's not available or chunked transfer encoding is used,
    ``None`` is returned.

    .. versionadded:: 0.9

    :param environ: the WSGI environ to fetch the content length from.
    """
    if environ.get("HTTP_TRANSFER_ENCODING", "") == "chunked":
        return None

    content_length = environ.get("CONTENT_LENGTH")
    if content_length is not None:
        try:
            return max(0, int(content_length))
        except (ValueError, TypeError):
            pass
    return None


def get_input_stream(
    environ: "WSGIEnvironment", safe_fallback: bool = True
) -> t.IO[bytes]:
    """Returns the input stream from the WSGI environment and wraps it
    in the most sensible way possible. The stream returned is not the
    raw WSGI stream in most cases but one that is safe to read from
    without taking into account the content length.

    If content length is not set, the stream will be empty for safety reasons.
    If the WSGI server supports chunked or infinite streams, it should set
    the ``wsgi.input_terminated`` value in the WSGI environ to indicate that.

    .. versionadded:: 0.9

    :param environ: the WSGI environ to fetch the stream from.
    :param safe_fallback: use an empty stream as a safe fallback when the
        content length is not set. Disabling this allows infinite streams,
        which can be a denial-of-service risk.
    """
    stream = t.cast(t.IO[bytes], environ["wsgi.input"])
    content_length = get_content_length(environ)

    # A wsgi extension that tells us if the input is terminated.  In
    # that case we return the stream unchanged as we know we can safely
    # read it until the end.
    if environ.get("wsgi.input_terminated"):
        return stream

    # If the request doesn't specify a content length, returning the stream is
    # potentially dangerous because it could be infinite, malicious or not. If
    # safe_fallback is true, return an empty stream instead for safety.
    if content_length is None:
        return io.BytesIO() if safe_fallback else stream

    # Otherwise limit the stream to the content length
    return t.cast(t.IO[bytes], LimitedStream(stream, content_length))


def get_query_string(environ: "WSGIEnvironment") -> str:
    """Returns the ``QUERY_STRING`` from the WSGI environment. This also
    takes care of the WSGI decoding dance. The string returned will be
    restricted to ASCII characters.

    :param environ: WSGI environment to get the query string from.

    .. versionadded:: 0.9
    """
    qs = environ.get("QUERY_STRING", "").encode("latin1")
    # QUERY_STRING really should be ascii safe but some browsers
    # will send us some unicode stuff (I am looking at you IE).
    # In that case we want to urllib quote it badly.
    return url_quote(qs, safe=":&%=+$!*'(),")


def get_path_info(
    environ: "WSGIEnvironment", charset: str = "utf-8", errors: str = "replace"
) -> str:
    """Return the ``PATH_INFO`` from the WSGI environment and decode it
    unless ``charset`` is ``None``.

    :param environ: WSGI environment to get the path from.
    :param charset: The charset for the path info, or ``None`` if no
        decoding should be performed.
    :param errors: The decoding error handling.

    .. versionadded:: 0.9
    """
    path = environ.get("PATH_INFO", "").encode("latin1")
    return _to_str(path, charset, errors, allow_none_charset=True)  # type: ignore


def get_script_name(
    environ: "WSGIEnvironment", charset: str = "utf-8", errors: str = "replace"
) -> str:
    """Return the ``SCRIPT_NAME`` from the WSGI environment and decode
    it unless `charset` is set to ``None``.

    :param environ: WSGI environment to get the path from.
    :param charset: The charset for the path, or ``None`` if no decoding
        should be performed.
    :param errors: The decoding error handling.

    .. versionadded:: 0.9
    """
    path = environ.get("SCRIPT_NAME", "").encode("latin1")
    return _to_str(path, charset, errors, allow_none_charset=True)  # type: ignore


def pop_path_info(
    environ: "WSGIEnvironment", charset: str = "utf-8", errors: str = "replace"
) -> t.Optional[str]:
    """Removes and returns the next segment of `PATH_INFO`, pushing it onto
    `SCRIPT_NAME`.  Returns `None` if there is nothing left on `PATH_INFO`.

    If the `charset` is set to `None` bytes are returned.

    If there are empty segments (``'/foo//bar``) these are ignored but
    properly pushed to the `SCRIPT_NAME`:

    >>> env = {'SCRIPT_NAME': '/foo', 'PATH_INFO': '/a/b'}
    >>> pop_path_info(env)
    'a'
    >>> env['SCRIPT_NAME']
    '/foo/a'
    >>> pop_path_info(env)
    'b'
    >>> env['SCRIPT_NAME']
    '/foo/a/b'

    .. versionadded:: 0.5

    .. versionchanged:: 0.9
       The path is now decoded and a charset and encoding
       parameter can be provided.

    :param environ: the WSGI environment that is modified.
    :param charset: The ``encoding`` parameter passed to
        :func:`bytes.decode`.
    :param errors: The ``errors`` paramater passed to
        :func:`bytes.decode`.
    """
    path = environ.get("PATH_INFO")
    if not path:
        return None

    script_name = environ.get("SCRIPT_NAME", "")

    # shift multiple leading slashes over
    old_path = path
    path = path.lstrip("/")
    if path != old_path:
        script_name += "/" * (len(old_path) - len(path))

    if "/" not in path:
        environ["PATH_INFO"] = ""
        environ["SCRIPT_NAME"] = script_name + path
        rv = path.encode("latin1")
    else:
        segment, path = path.split("/", 1)
        environ["PATH_INFO"] = f"/{path}"
        environ["SCRIPT_NAME"] = script_name + segment
        rv = segment.encode("latin1")

    return _to_str(rv, charset, errors, allow_none_charset=True)  # type: ignore


def peek_path_info(
    environ: "WSGIEnvironment", charset: str = "utf-8", errors: str = "replace"
) -> t.Optional[str]:
    """Returns the next segment on the `PATH_INFO` or `None` if there
    is none.  Works like :func:`pop_path_info` without modifying the
    environment:

    >>> env = {'SCRIPT_NAME': '/foo', 'PATH_INFO': '/a/b'}
    >>> peek_path_info(env)
    'a'
    >>> peek_path_info(env)
    'a'

    If the `charset` is set to `None` bytes are returned.

    .. versionadded:: 0.5

    .. versionchanged:: 0.9
       The path is now decoded and a charset and encoding
       parameter can be provided.

    :param environ: the WSGI environment that is checked.
    """
    segments = environ.get("PATH_INFO", "").lstrip("/").split("/", 1)
    if segments:
        return _to_str(  # type: ignore
            segments[0].encode("latin1"), charset, errors, allow_none_charset=True
        )
    return None


def extract_path_info(
    environ_or_baseurl: t.Union[str, "WSGIEnvironment"],
    path_or_url: t.Union[str, _URLTuple],
    charset: str = "utf-8",
    errors: str = "werkzeug.url_quote",
    collapse_http_schemes: bool = True,
) -> t.Optional[str]:
    """Extracts the path info from the given URL (or WSGI environment) and
    path. The path info returned is a string. The URLs might also be IRIs.

    If the path info could not be determined, `None` is returned.

    Some examples:

    >>> extract_path_info('http://example.com/app', '/app/hello')
    '/hello'
    >>> extract_path_info('http://example.com/app',
    ...                   'https://example.com/app/hello')
    '/hello'
    >>> extract_path_info('http://example.com/app',
    ...                   'https://example.com/app/hello',
    ...                   collapse_http_schemes=False) is None
    True

    Instead of providing a base URL you can also pass a WSGI environment.

    :param environ_or_baseurl: a WSGI environment dict, a base URL or
                               base IRI.  This is the root of the
                               application.
    :param path_or_url: an absolute path from the server root, a
                        relative path (in which case it's the path info)
                        or a full URL.
    :param charset: the charset for byte data in URLs
    :param errors: the error handling on decode
    :param collapse_http_schemes: if set to `False` the algorithm does
                                  not assume that http and https on the
                                  same server point to the same
                                  resource.

    .. versionchanged:: 0.15
        The ``errors`` parameter defaults to leaving invalid bytes
        quoted instead of replacing them.

    .. versionadded:: 0.6
    """

    def _normalize_netloc(scheme: str, netloc: str) -> str:
        parts = netloc.split("@", 1)[-1].split(":", 1)
        port: t.Optional[str]

        if len(parts) == 2:
            netloc, port = parts
            if (scheme == "http" and port == "80") or (
                scheme == "https" and port == "443"
            ):
                port = None
        else:
            netloc = parts[0]
            port = None

        if port is not None:
            netloc += f":{port}"

        return netloc

    # make sure whatever we are working on is a IRI and parse it
    path = uri_to_iri(path_or_url, charset, errors)
    if isinstance(environ_or_baseurl, dict):
        environ_or_baseurl = get_current_url(environ_or_baseurl, root_only=True)
    base_iri = uri_to_iri(environ_or_baseurl, charset, errors)
    base_scheme, base_netloc, base_path = url_parse(base_iri)[:3]
    cur_scheme, cur_netloc, cur_path = url_parse(url_join(base_iri, path))[:3]

    # normalize the network location
    base_netloc = _normalize_netloc(base_scheme, base_netloc)
    cur_netloc = _normalize_netloc(cur_scheme, cur_netloc)

    # is that IRI even on a known HTTP scheme?
    if collapse_http_schemes:
        for scheme in base_scheme, cur_scheme:
            if scheme not in ("http", "https"):
                return None
    else:
        if not (base_scheme in ("http", "https") and base_scheme == cur_scheme):
            return None

    # are the netlocs compatible?
    if base_netloc != cur_netloc:
        return None

    # are we below the application path?
    base_path = base_path.rstrip("/")
    if not cur_path.startswith(base_path):
        return None

    return f"/{cur_path[len(base_path) :].lstrip('/')}"


class ClosingIterator:
    """The WSGI specification requires that all middlewares and gateways
    respect the `close` callback of the iterable returned by the application.
    Because it is useful to add another close action to a returned iterable
    and adding a custom iterable is a boring task this class can be used for
    that::

        return ClosingIterator(app(environ, start_response), [cleanup_session,
                                                              cleanup_locals])

    If there is just one close function it can be passed instead of the list.

    A closing iterator is not needed if the application uses response objects
    and finishes the processing if the response is started::

        try:
            return response(environ, start_response)
        finally:
            cleanup_session()
            cleanup_locals()
    """

    def __init__(
        self,
        iterable: t.Iterable[bytes],
        callbacks: t.Optional[
            t.Union[t.Callable[[], None], t.Iterable[t.Callable[[], None]]]
        ] = None,
    ) -> None:
        iterator = iter(iterable)
        self._next = t.cast(t.Callable[[], bytes], partial(next, iterator))
        if callbacks is None:
            callbacks = []
        elif callable(callbacks):
            callbacks = [callbacks]
        else:
            callbacks = list(callbacks)
        iterable_close = getattr(iterable, "close", None)
        if iterable_close:
            callbacks.insert(0, iterable_close)
        self._callbacks = callbacks

    def __iter__(self) -> "ClosingIterator":
        return self

    def __next__(self) -> bytes:
        return self._next()

    def close(self) -> None:
        for callback in self._callbacks:
            callback()


def wrap_file(
    environ: "WSGIEnvironment", file: t.IO[bytes], buffer_size: int = 8192
) -> t.Iterable[bytes]:
    """Wraps a file.  This uses the WSGI server's file wrapper if available
    or otherwise the generic :class:`FileWrapper`.

    .. versionadded:: 0.5

    If the file wrapper from the WSGI server is used it's important to not
    iterate over it from inside the application but to pass it through
    unchanged.  If you want to pass out a file wrapper inside a response
    object you have to set :attr:`Response.direct_passthrough` to `True`.

    More information about file wrappers are available in :pep:`333`.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """
    return environ.get("wsgi.file_wrapper", FileWrapper)(  # type: ignore
        file, buffer_size
    )


class FileWrapper:
    """This class can be used to convert a :class:`file`-like object into
    an iterable.  It yields `buffer_size` blocks until the file is fully
    read.

    You should not use this class directly but rather use the
    :func:`wrap_file` function that uses the WSGI server's file wrapper
    support if it's available.

    .. versionadded:: 0.5

    If you're using this object together with a :class:`Response` you have
    to use the `direct_passthrough` mode.

    :param file: a :class:`file`-like object with a :meth:`~file.read` method.
    :param buffer_size: number of bytes for one iteration.
    """

    def __init__(self, file: t.IO[bytes], buffer_size: int = 8192) -> None:
        self.file = file
        self.buffer_size = buffer_size

    def close(self) -> None:
        if hasattr(self.file, "close"):
            self.file.close()

    def seekable(self) -> bool:
        if hasattr(self.file, "seekable"):
            return self.file.seekable()
        if hasattr(self.file, "seek"):
            return True
        return False

    def seek(self, *args: t.Any) -> None:
        if hasattr(self.file, "seek"):
            self.file.seek(*args)

    def tell(self) -> t.Optional[int]:
        if hasattr(self.file, "tell"):
            return self.file.tell()
        return None

    def __iter__(self) -> "FileWrapper":
        return self

    def __next__(self) -> bytes:
        data = self.file.read(self.buffer_size)
        if data:
            return data
        raise StopIteration()


class _RangeWrapper:
    # private for now, but should we make it public in the future ?

    """This class can be used to convert an iterable object into
    an iterable that will only yield a piece of the underlying content.
    It yields blocks until the underlying stream range is fully read.
    The yielded blocks will have a size that can't exceed the original
    iterator defined block size, but that can be smaller.

    If you're using this object together with a :class:`Response` you have
    to use the `direct_passthrough` mode.

    :param iterable: an iterable object with a :meth:`__next__` method.
    :param start_byte: byte from which read will start.
    :param byte_range: how many bytes to read.
    """

    def __init__(
        self,
        iterable: t.Union[t.Iterable[bytes], t.IO[bytes]],
        start_byte: int = 0,
        byte_range: t.Optional[int] = None,
    ):
        self.iterable = iter(iterable)
        self.byte_range = byte_range
        self.start_byte = start_byte
        self.end_byte = None

        if byte_range is not None:
            self.end_byte = start_byte + byte_range

        self.read_length = 0
        self.seekable = (
            hasattr(iterable, "seekable") and iterable.seekable()  # type: ignore
        )
        self.end_reached = False

    def __iter__(self) -> "_RangeWrapper":
        return self

    def _next_chunk(self) -> bytes:
        try:
            chunk = next(self.iterable)
            self.read_length += len(chunk)
            return chunk
        except StopIteration:
            self.end_reached = True
            raise

    def _first_iteration(self) -> t.Tuple[t.Optional[bytes], int]:
        chunk = None
        if self.seekable:
            self.iterable.seek(self.start_byte)  # type: ignore
            self.read_length = self.iterable.tell()  # type: ignore
            contextual_read_length = self.read_length
        else:
            while self.read_length <= self.start_byte:
                chunk = self._next_chunk()
            if chunk is not None:
                chunk = chunk[self.start_byte - self.read_length :]
            contextual_read_length = self.start_byte
        return chunk, contextual_read_length

    def _next(self) -> bytes:
        if self.end_reached:
            raise StopIteration()
        chunk = None
        contextual_read_length = self.read_length
        if self.read_length == 0:
            chunk, contextual_read_length = self._first_iteration()
        if chunk is None:
            chunk = self._next_chunk()
        if self.end_byte is not None and self.read_length >= self.end_byte:
            self.end_reached = True
            return chunk[: self.end_byte - contextual_read_length]
        return chunk

    def __next__(self) -> bytes:
        chunk = self._next()
        if chunk:
            return chunk
        self.end_reached = True
        raise StopIteration()

    def close(self) -> None:
        if hasattr(self.iterable, "close"):
            self.iterable.close()  # type: ignore


def _make_chunk_iter(
    stream: t.Union[t.Iterable[bytes], t.IO[bytes]],
    limit: t.Optional[int],
    buffer_size: int,
) -> t.Iterator[bytes]:
    """Helper for the line and chunk iter functions."""
    if isinstance(stream, (bytes, bytearray, str)):
        raise TypeError(
            "Passed a string or byte object instead of true iterator or stream."
        )
    if not hasattr(stream, "read"):
        for item in stream:
            if item:
                yield item
        return
    stream = t.cast(t.IO[bytes], stream)
    if not isinstance(stream, LimitedStream) and limit is not None:
        stream = t.cast(t.IO[bytes], LimitedStream(stream, limit))
    _read = stream.read
    while True:
        item = _read(buffer_size)
        if not item:
            break
        yield item


def make_line_iter(
    stream: t.Union[t.Iterable[bytes], t.IO[bytes]],
    limit: t.Optional[int] = None,
    buffer_size: int = 10 * 1024,
    cap_at_buffer: bool = False,
) -> t.Iterator[bytes]:
    """Safely iterates line-based over an input stream.  If the input stream
    is not a :class:`LimitedStream` the `limit` parameter is mandatory.

    This uses the stream's :meth:`~file.read` method internally as opposite
    to the :meth:`~file.readline` method that is unsafe and can only be used
    in violation of the WSGI specification.  The same problem applies to the
    `__iter__` function of the input stream which calls :meth:`~file.readline`
    without arguments.

    If you need line-by-line processing it's strongly recommended to iterate
    over the input stream using this helper function.

    .. versionchanged:: 0.8
       This function now ensures that the limit was reached.

    .. versionadded:: 0.9
       added support for iterators as input stream.

    .. versionadded:: 0.11.10
       added support for the `cap_at_buffer` parameter.

    :param stream: the stream or iterate to iterate over.
    :param limit: the limit in bytes for the stream.  (Usually
                  content length.  Not necessary if the `stream`
                  is a :class:`LimitedStream`.
    :param buffer_size: The optional buffer size.
    :param cap_at_buffer: if this is set chunks are split if they are longer
                          than the buffer size.  Internally this is implemented
                          that the buffer size might be exhausted by a factor
                          of two however.
    """
    _iter = _make_chunk_iter(stream, limit, buffer_size)

    first_item = next(_iter, "")
    if not first_item:
        return

    s = _make_encode_wrapper(first_item)
    empty = t.cast(bytes, s(""))
    cr = t.cast(bytes, s("\r"))
    lf = t.cast(bytes, s("\n"))
    crlf = t.cast(bytes, s("\r\n"))

    _iter = t.cast(t.Iterator[bytes], chain((first_item,), _iter))

    def _iter_basic_lines() -> t.Iterator[bytes]:
        _join = empty.join
        buffer: t.List[bytes] = []
        while True:
            new_data = next(_iter, "")
            if not new_data:
                break
            new_buf: t.List[bytes] = []
            buf_size = 0
            for item in t.cast(
                t.Iterator[bytes], chain(buffer, new_data.splitlines(True))
            ):
                new_buf.append(item)
                buf_size += len(item)
                if item and item[-1:] in crlf:
                    yield _join(new_buf)
                    new_buf = []
                elif cap_at_buffer and buf_size >= buffer_size:
                    rv = _join(new_buf)
                    while len(rv) >= buffer_size:
                        yield rv[:buffer_size]
                        rv = rv[buffer_size:]
                    new_buf = [rv]
            buffer = new_buf
        if buffer:
            yield _join(buffer)

    # This hackery is necessary to merge 'foo\r' and '\n' into one item
    # of 'foo\r\n' if we were unlucky and we hit a chunk boundary.
    previous = empty
    for item in _iter_basic_lines():
        if item == lf and previous[-1:] == cr:
            previous += item
            item = empty
        if previous:
            yield previous
        previous = item
    if previous:
        yield previous


def make_chunk_iter(
    stream: t.Union[t.Iterable[bytes], t.IO[bytes]],
    separator: bytes,
    limit: t.Optional[int] = None,
    buffer_size: int = 10 * 1024,
    cap_at_buffer: bool = False,
) -> t.Iterator[bytes]:
    """Works like :func:`make_line_iter` but accepts a separator
    which divides chunks.  If you want newline based processing
    you should use :func:`make_line_iter` instead as it
    supports arbitrary newline markers.

    .. versionadded:: 0.8

    .. versionadded:: 0.9
       added support for iterators as input stream.

    .. versionadded:: 0.11.10
       added support for the `cap_at_buffer` parameter.

    :param stream: the stream or iterate to iterate over.
    :param separator: the separator that divides chunks.
    :param limit: the limit in bytes for the stream.  (Usually
                  content length.  Not necessary if the `stream`
                  is otherwise already limited).
    :param buffer_size: The optional buffer size.
    :param cap_at_buffer: if this is set chunks are split if they are longer
                          than the buffer size.  Internally this is implemented
                          that the buffer size might be exhausted by a factor
                          of two however.
    """
    _iter = _make_chunk_iter(stream, limit, buffer_size)

    first_item = next(_iter, b"")
    if not first_item:
        return

    _iter = t.cast(t.Iterator[bytes], chain((first_item,), _iter))
    if isinstance(first_item, str):
        separator = _to_str(separator)
        _split = re.compile(f"({re.escape(separator)})").split
        _join = "".join
    else:
        separator = _to_bytes(separator)
        _split = re.compile(b"(" + re.escape(separator) + b")").split
        _join = b"".join

    buffer: t.List[bytes] = []
    while True:
        new_data = next(_iter, b"")
        if not new_data:
            break
        chunks = _split(new_data)
        new_buf: t.List[bytes] = []
        buf_size = 0
        for item in chain(buffer, chunks):
            if item == separator:
                yield _join(new_buf)
                new_buf = []
                buf_size = 0
            else:
                buf_size += len(item)
                new_buf.append(item)

                if cap_at_buffer and buf_size >= buffer_size:
                    rv = _join(new_buf)
                    while len(rv) >= buffer_size:
                        yield rv[:buffer_size]
                        rv = rv[buffer_size:]
                    new_buf = [rv]
                    buf_size = len(rv)

        buffer = new_buf
    if buffer:
        yield _join(buffer)


class LimitedStream(io.IOBase):
    """Wraps a stream so that it doesn't read more than n bytes.  If the
    stream is exhausted and the caller tries to get more bytes from it
    :func:`on_exhausted` is called which by default returns an empty
    string.  The return value of that function is forwarded
    to the reader function.  So if it returns an empty string
    :meth:`read` will return an empty string as well.

    The limit however must never be higher than what the stream can
    output.  Otherwise :meth:`readlines` will try to read past the
    limit.

    .. admonition:: Note on WSGI compliance

       calls to :meth:`readline` and :meth:`readlines` are not
       WSGI compliant because it passes a size argument to the
       readline methods.  Unfortunately the WSGI PEP is not safely
       implementable without a size argument to :meth:`readline`
       because there is no EOF marker in the stream.  As a result
       of that the use of :meth:`readline` is discouraged.

       For the same reason iterating over the :class:`LimitedStream`
       is not portable.  It internally calls :meth:`readline`.

       We strongly suggest using :meth:`read` only or using the
       :func:`make_line_iter` which safely iterates line-based
       over a WSGI input stream.

    :param stream: the stream to wrap.
    :param limit: the limit for the stream, must not be longer than
                  what the string can provide if the stream does not
                  end with `EOF` (like `wsgi.input`)
    """

    def __init__(self, stream: t.IO[bytes], limit: int) -> None:
        self._read = stream.read
        self._readline = stream.readline
        self._pos = 0
        self.limit = limit

    def __iter__(self) -> "LimitedStream":
        return self

    @property
    def is_exhausted(self) -> bool:
        """If the stream is exhausted this attribute is `True`."""
        return self._pos >= self.limit

    def on_exhausted(self) -> bytes:
        """This is called when the stream tries to read past the limit.
        The return value of this function is returned from the reading
        function.
        """
        # Read null bytes from the stream so that we get the
        # correct end of stream marker.
        return self._read(0)

    def on_disconnect(self) -> bytes:
        """What should happen if a disconnect is detected?  The return
        value of this function is returned from read functions in case
        the client went away.  By default a
        :exc:`~werkzeug.exceptions.ClientDisconnected` exception is raised.
        """
        from .exceptions import ClientDisconnected

        raise ClientDisconnected()

    def exhaust(self, chunk_size: int = 1024 * 64) -> None:
        """Exhaust the stream.  This consumes all the data left until the
        limit is reached.

        :param chunk_size: the size for a chunk.  It will read the chunk
                           until the stream is exhausted and throw away
                           the results.
        """
        to_read = self.limit - self._pos
        chunk = chunk_size
        while to_read > 0:
            chunk = min(to_read, chunk)
            self.read(chunk)
            to_read -= chunk

    def read(self, size: t.Optional[int] = None) -> bytes:
        """Read `size` bytes or if size is not provided everything is read.

        :param size: the number of bytes read.
        """
        if self._pos >= self.limit:
            return self.on_exhausted()
        if size is None or size == -1:  # -1 is for consistence with file
            size = self.limit
        to_read = min(self.limit - self._pos, size)
        try:
            read = self._read(to_read)
        except (OSError, ValueError):
            return self.on_disconnect()
        if to_read and len(read) != to_read:
            return self.on_disconnect()
        self._pos += len(read)
        return read

    def readline(self, size: t.Optional[int] = None) -> bytes:
        """Reads one line from the stream."""
        if self._pos >= self.limit:
            return self.on_exhausted()
        if size is None:
            size = self.limit - self._pos
        else:
            size = min(size, self.limit - self._pos)
        try:
            line = self._readline(size)
        except (ValueError, OSError):
            return self.on_disconnect()
        if size and not line:
            return self.on_disconnect()
        self._pos += len(line)
        return line

    def readlines(self, size: t.Optional[int] = None) -> t.List[bytes]:
        """Reads a file into a list of strings.  It calls :meth:`readline`
        until the file is read to the end.  It does support the optional
        `size` argument if the underlying stream supports it for
        `readline`.
        """
        last_pos = self._pos
        result = []
        if size is not None:
            end = min(self.limit, last_pos + size)
        else:
            end = self.limit
        while True:
            if size is not None:
                size -= last_pos - self._pos
            if self._pos >= end:
                break
            result.append(self.readline(size))
            if size is not None:
                last_pos = self._pos
        return result

    def tell(self) -> int:
        """Returns the position of the stream.

        .. versionadded:: 0.9
        """
        return self._pos

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration()
        return line

    def readable(self) -> bool:
        return True
