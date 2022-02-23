import typing as t
import warnings
from functools import update_wrapper
from io import BytesIO
from itertools import chain
from typing import Union

from . import exceptions
from ._internal import _to_str
from .datastructures import FileStorage
from .datastructures import Headers
from .datastructures import MultiDict
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartDecoder
from .sansio.multipart import NeedData
from .urls import url_decode_stream
from .wsgi import _make_chunk_iter
from .wsgi import get_content_length
from .wsgi import get_input_stream

# there are some platforms where SpooledTemporaryFile is not available.
# In that case we need to provide a fallback.
try:
    from tempfile import SpooledTemporaryFile
except ImportError:
    from tempfile import TemporaryFile

    SpooledTemporaryFile = None  # type: ignore

if t.TYPE_CHECKING:
    import typing as te
    from _typeshed.wsgi import WSGIEnvironment

    t_parse_result = t.Tuple[t.IO[bytes], MultiDict, MultiDict]

    class TStreamFactory(te.Protocol):
        def __call__(
            self,
            total_content_length: t.Optional[int],
            content_type: t.Optional[str],
            filename: t.Optional[str],
            content_length: t.Optional[int] = None,
        ) -> t.IO[bytes]:
            ...


F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def _exhaust(stream: t.IO[bytes]) -> None:
    bts = stream.read(64 * 1024)
    while bts:
        bts = stream.read(64 * 1024)


def default_stream_factory(
    total_content_length: t.Optional[int],
    content_type: t.Optional[str],
    filename: t.Optional[str],
    content_length: t.Optional[int] = None,
) -> t.IO[bytes]:
    max_size = 1024 * 500

    if SpooledTemporaryFile is not None:
        return t.cast(t.IO[bytes], SpooledTemporaryFile(max_size=max_size, mode="rb+"))
    elif total_content_length is None or total_content_length > max_size:
        return t.cast(t.IO[bytes], TemporaryFile("rb+"))

    return BytesIO()


def parse_form_data(
    environ: "WSGIEnvironment",
    stream_factory: t.Optional["TStreamFactory"] = None,
    charset: str = "utf-8",
    errors: str = "replace",
    max_form_memory_size: t.Optional[int] = None,
    max_content_length: t.Optional[int] = None,
    cls: t.Optional[t.Type[MultiDict]] = None,
    silent: bool = True,
) -> "t_parse_result":
    """Parse the form data in the environ and return it as tuple in the form
    ``(stream, form, files)``.  You should only call this method if the
    transport method is `POST`, `PUT`, or `PATCH`.

    If the mimetype of the data transmitted is `multipart/form-data` the
    files multidict will be filled with `FileStorage` objects.  If the
    mimetype is unknown the input stream is wrapped and returned as first
    argument, else the stream is empty.

    This is a shortcut for the common usage of :class:`FormDataParser`.

    Have a look at :doc:`/request_data` for more details.

    .. versionadded:: 0.5
       The `max_form_memory_size`, `max_content_length` and
       `cls` parameters were added.

    .. versionadded:: 0.5.1
       The optional `silent` flag was added.

    :param environ: the WSGI environment to be used for parsing.
    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`Response._get_file_stream`.
    :param charset: The character set for URL and url encoded form data.
    :param errors: The encoding error behavior.
    :param max_form_memory_size: the maximum number of bytes to be accepted for
                           in-memory stored form data.  If the data
                           exceeds the value specified an
                           :exc:`~exceptions.RequestEntityTooLarge`
                           exception is raised.
    :param max_content_length: If this is provided and the transmitted data
                               is longer than this value an
                               :exc:`~exceptions.RequestEntityTooLarge`
                               exception is raised.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    :return: A tuple in the form ``(stream, form, files)``.
    """
    return FormDataParser(
        stream_factory,
        charset,
        errors,
        max_form_memory_size,
        max_content_length,
        cls,
        silent,
    ).parse_from_environ(environ)


def exhaust_stream(f: F) -> F:
    """Helper decorator for methods that exhausts the stream on return."""

    def wrapper(self, stream, *args, **kwargs):  # type: ignore
        try:
            return f(self, stream, *args, **kwargs)
        finally:
            exhaust = getattr(stream, "exhaust", None)

            if exhaust is not None:
                exhaust()
            else:
                while True:
                    chunk = stream.read(1024 * 64)

                    if not chunk:
                        break

    return update_wrapper(t.cast(F, wrapper), f)


class FormDataParser:
    """This class implements parsing of form data for Werkzeug.  By itself
    it can parse multipart and url encoded form data.  It can be subclassed
    and extended but for most mimetypes it is a better idea to use the
    untouched stream and expose it as separate attributes on a request
    object.

    .. versionadded:: 0.8

    :param stream_factory: An optional callable that returns a new read and
                           writeable file descriptor.  This callable works
                           the same as :meth:`Response._get_file_stream`.
    :param charset: The character set for URL and url encoded form data.
    :param errors: The encoding error behavior.
    :param max_form_memory_size: the maximum number of bytes to be accepted for
                           in-memory stored form data.  If the data
                           exceeds the value specified an
                           :exc:`~exceptions.RequestEntityTooLarge`
                           exception is raised.
    :param max_content_length: If this is provided and the transmitted data
                               is longer than this value an
                               :exc:`~exceptions.RequestEntityTooLarge`
                               exception is raised.
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param silent: If set to False parsing errors will not be caught.
    """

    def __init__(
        self,
        stream_factory: t.Optional["TStreamFactory"] = None,
        charset: str = "utf-8",
        errors: str = "replace",
        max_form_memory_size: t.Optional[int] = None,
        max_content_length: t.Optional[int] = None,
        cls: t.Optional[t.Type[MultiDict]] = None,
        silent: bool = True,
    ) -> None:
        if stream_factory is None:
            stream_factory = default_stream_factory

        self.stream_factory = stream_factory
        self.charset = charset
        self.errors = errors
        self.max_form_memory_size = max_form_memory_size
        self.max_content_length = max_content_length

        if cls is None:
            cls = MultiDict

        self.cls = cls
        self.silent = silent

    def get_parse_func(
        self, mimetype: str, options: t.Dict[str, str]
    ) -> t.Optional[
        t.Callable[
            ["FormDataParser", t.IO[bytes], str, t.Optional[int], t.Dict[str, str]],
            "t_parse_result",
        ]
    ]:
        return self.parse_functions.get(mimetype)

    def parse_from_environ(self, environ: "WSGIEnvironment") -> "t_parse_result":
        """Parses the information from the environment as form data.

        :param environ: the WSGI environment to be used for parsing.
        :return: A tuple in the form ``(stream, form, files)``.
        """
        content_type = environ.get("CONTENT_TYPE", "")
        content_length = get_content_length(environ)
        mimetype, options = parse_options_header(content_type)
        return self.parse(get_input_stream(environ), mimetype, content_length, options)

    def parse(
        self,
        stream: t.IO[bytes],
        mimetype: str,
        content_length: t.Optional[int],
        options: t.Optional[t.Dict[str, str]] = None,
    ) -> "t_parse_result":
        """Parses the information from the given stream, mimetype,
        content length and mimetype parameters.

        :param stream: an input stream
        :param mimetype: the mimetype of the data
        :param content_length: the content length of the incoming data
        :param options: optional mimetype parameters (used for
                        the multipart boundary for instance)
        :return: A tuple in the form ``(stream, form, files)``.
        """
        if (
            self.max_content_length is not None
            and content_length is not None
            and content_length > self.max_content_length
        ):
            # if the input stream is not exhausted, firefox reports Connection Reset
            _exhaust(stream)
            raise exceptions.RequestEntityTooLarge()

        if options is None:
            options = {}

        parse_func = self.get_parse_func(mimetype, options)

        if parse_func is not None:
            try:
                return parse_func(self, stream, mimetype, content_length, options)
            except ValueError:
                if not self.silent:
                    raise

        return stream, self.cls(), self.cls()

    @exhaust_stream
    def _parse_multipart(
        self,
        stream: t.IO[bytes],
        mimetype: str,
        content_length: t.Optional[int],
        options: t.Dict[str, str],
    ) -> "t_parse_result":
        parser = MultiPartParser(
            self.stream_factory,
            self.charset,
            self.errors,
            max_form_memory_size=self.max_form_memory_size,
            cls=self.cls,
        )
        boundary = options.get("boundary", "").encode("ascii")

        if not boundary:
            raise ValueError("Missing boundary")

        form, files = parser.parse(stream, boundary, content_length)
        return stream, form, files

    @exhaust_stream
    def _parse_urlencoded(
        self,
        stream: t.IO[bytes],
        mimetype: str,
        content_length: t.Optional[int],
        options: t.Dict[str, str],
    ) -> "t_parse_result":
        if (
            self.max_form_memory_size is not None
            and content_length is not None
            and content_length > self.max_form_memory_size
        ):
            # if the input stream is not exhausted, firefox reports Connection Reset
            _exhaust(stream)
            raise exceptions.RequestEntityTooLarge()

        form = url_decode_stream(stream, self.charset, errors=self.errors, cls=self.cls)
        return stream, form, self.cls()

    #: mapping of mimetypes to parsing functions
    parse_functions: t.Dict[
        str,
        t.Callable[
            ["FormDataParser", t.IO[bytes], str, t.Optional[int], t.Dict[str, str]],
            "t_parse_result",
        ],
    ] = {
        "multipart/form-data": _parse_multipart,
        "application/x-www-form-urlencoded": _parse_urlencoded,
        "application/x-url-encoded": _parse_urlencoded,
    }


def _line_parse(line: str) -> t.Tuple[str, bool]:
    """Removes line ending characters and returns a tuple (`stripped_line`,
    `is_terminated`).
    """
    if line[-2:] == "\r\n":
        return line[:-2], True

    elif line[-1:] in {"\r", "\n"}:
        return line[:-1], True

    return line, False


def parse_multipart_headers(iterable: t.Iterable[bytes]) -> Headers:
    """Parses multipart headers from an iterable that yields lines (including
    the trailing newline symbol).  The iterable has to be newline terminated.
    The iterable will stop at the line where the headers ended so it can be
    further consumed.
    :param iterable: iterable of strings that are newline terminated
    """
    warnings.warn(
        "'parse_multipart_headers' is deprecated and will be removed in"
        " Werkzeug 2.1.",
        DeprecationWarning,
        stacklevel=2,
    )
    result: t.List[t.Tuple[str, str]] = []

    for b_line in iterable:
        line = _to_str(b_line)
        line, line_terminated = _line_parse(line)

        if not line_terminated:
            raise ValueError("unexpected end of line in multipart header")

        if not line:
            break
        elif line[0] in " \t" and result:
            key, value = result[-1]
            result[-1] = (key, f"{value}\n {line[1:]}")
        else:
            parts = line.split(":", 1)

            if len(parts) == 2:
                result.append((parts[0].strip(), parts[1].strip()))

    # we link the list to the headers, no need to create a copy, the
    # list was not shared anyways.
    return Headers(result)


class MultiPartParser:
    def __init__(
        self,
        stream_factory: t.Optional["TStreamFactory"] = None,
        charset: str = "utf-8",
        errors: str = "replace",
        max_form_memory_size: t.Optional[int] = None,
        cls: t.Optional[t.Type[MultiDict]] = None,
        buffer_size: int = 64 * 1024,
    ) -> None:
        self.charset = charset
        self.errors = errors
        self.max_form_memory_size = max_form_memory_size

        if stream_factory is None:
            stream_factory = default_stream_factory

        self.stream_factory = stream_factory

        if cls is None:
            cls = MultiDict

        self.cls = cls

        self.buffer_size = buffer_size

    def fail(self, message: str) -> "te.NoReturn":
        raise ValueError(message)

    def get_part_charset(self, headers: Headers) -> str:
        # Figure out input charset for current part
        content_type = headers.get("content-type")

        if content_type:
            mimetype, ct_params = parse_options_header(content_type)
            return ct_params.get("charset", self.charset)

        return self.charset

    def start_file_streaming(
        self, event: File, total_content_length: t.Optional[int]
    ) -> t.IO[bytes]:
        content_type = event.headers.get("content-type")

        try:
            content_length = int(event.headers["content-length"])
        except (KeyError, ValueError):
            content_length = 0

        container = self.stream_factory(
            total_content_length=total_content_length,
            filename=event.filename,
            content_type=content_type,
            content_length=content_length,
        )
        return container

    def parse(
        self, stream: t.IO[bytes], boundary: bytes, content_length: t.Optional[int]
    ) -> t.Tuple[MultiDict, MultiDict]:
        container: t.Union[t.IO[bytes], t.List[bytes]]
        _write: t.Callable[[bytes], t.Any]

        iterator = chain(
            _make_chunk_iter(
                stream,
                limit=content_length,
                buffer_size=self.buffer_size,
            ),
            [None],
        )

        parser = MultipartDecoder(boundary, self.max_form_memory_size)

        fields = []
        files = []

        current_part: Union[Field, File]
        for data in iterator:
            parser.receive_data(data)
            event = parser.next_event()
            while not isinstance(event, (Epilogue, NeedData)):
                if isinstance(event, Field):
                    current_part = event
                    container = []
                    _write = container.append
                elif isinstance(event, File):
                    current_part = event
                    container = self.start_file_streaming(event, content_length)
                    _write = container.write
                elif isinstance(event, Data):
                    _write(event.data)
                    if not event.more_data:
                        if isinstance(current_part, Field):
                            value = b"".join(container).decode(
                                self.get_part_charset(current_part.headers), self.errors
                            )
                            fields.append((current_part.name, value))
                        else:
                            container = t.cast(t.IO[bytes], container)
                            container.seek(0)
                            files.append(
                                (
                                    current_part.name,
                                    FileStorage(
                                        container,
                                        current_part.filename,
                                        current_part.name,
                                        headers=current_part.headers,
                                    ),
                                )
                            )

                event = parser.next_event()

        return self.cls(fields), self.cls(files)
