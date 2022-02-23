import mimetypes
import sys
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from http.cookiejar import CookieJar
from io import BytesIO
from itertools import chain
from random import random
from tempfile import TemporaryFile
from time import time
from urllib.request import Request as _UrllibRequest

from ._internal import _get_environ
from ._internal import _make_encode_wrapper
from ._internal import _wsgi_decoding_dance
from ._internal import _wsgi_encoding_dance
from .datastructures import Authorization
from .datastructures import CallbackDict
from .datastructures import CombinedMultiDict
from .datastructures import EnvironHeaders
from .datastructures import FileMultiDict
from .datastructures import Headers
from .datastructures import MultiDict
from .http import dump_cookie
from .http import dump_options_header
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartEncoder
from .sansio.multipart import Preamble
from .urls import iri_to_uri
from .urls import url_encode
from .urls import url_fix
from .urls import url_parse
from .urls import url_unparse
from .urls import url_unquote
from .utils import get_content_type
from .wrappers.request import Request
from .wrappers.response import Response
from .wsgi import ClosingIterator
from .wsgi import get_current_url

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


def stream_encode_multipart(
    data: t.Mapping[str, t.Any],
    use_tempfile: bool = True,
    threshold: int = 1024 * 500,
    boundary: t.Optional[str] = None,
    charset: str = "utf-8",
) -> t.Tuple[t.IO[bytes], int, str]:
    """Encode a dict of values (either strings or file descriptors or
    :class:`FileStorage` objects.) into a multipart encoded string stored
    in a file descriptor.
    """
    if boundary is None:
        boundary = f"---------------WerkzeugFormPart_{time()}{random()}"

    stream: t.IO[bytes] = BytesIO()
    total_length = 0
    on_disk = False

    if use_tempfile:

        def write_binary(s: bytes) -> int:
            nonlocal stream, total_length, on_disk

            if on_disk:
                return stream.write(s)
            else:
                length = len(s)

                if length + total_length <= threshold:
                    stream.write(s)
                else:
                    new_stream = t.cast(t.IO[bytes], TemporaryFile("wb+"))
                    new_stream.write(stream.getvalue())  # type: ignore
                    new_stream.write(s)
                    stream = new_stream
                    on_disk = True

                total_length += length
                return length

    else:
        write_binary = stream.write

    encoder = MultipartEncoder(boundary.encode())
    write_binary(encoder.send_event(Preamble(data=b"")))
    for key, value in _iter_data(data):
        reader = getattr(value, "read", None)
        if reader is not None:
            filename = getattr(value, "filename", getattr(value, "name", None))
            content_type = getattr(value, "content_type", None)
            if content_type is None:
                content_type = (
                    filename
                    and mimetypes.guess_type(filename)[0]
                    or "application/octet-stream"
                )
            headers = Headers([("Content-Type", content_type)])
            if filename is None:
                write_binary(encoder.send_event(Field(name=key, headers=headers)))
            else:
                write_binary(
                    encoder.send_event(
                        File(name=key, filename=filename, headers=headers)
                    )
                )
            while True:
                chunk = reader(16384)

                if not chunk:
                    break

                write_binary(encoder.send_event(Data(data=chunk, more_data=True)))
        else:
            if not isinstance(value, str):
                value = str(value)
            write_binary(encoder.send_event(Field(name=key, headers=Headers())))
            write_binary(
                encoder.send_event(Data(data=value.encode(charset), more_data=False))
            )

    write_binary(encoder.send_event(Epilogue(data=b"")))

    length = stream.tell()
    stream.seek(0)
    return stream, length, boundary


def encode_multipart(
    values: t.Mapping[str, t.Any],
    boundary: t.Optional[str] = None,
    charset: str = "utf-8",
) -> t.Tuple[str, bytes]:
    """Like `stream_encode_multipart` but returns a tuple in the form
    (``boundary``, ``data``) where data is bytes.
    """
    stream, length, boundary = stream_encode_multipart(
        values, use_tempfile=False, boundary=boundary, charset=charset
    )
    return boundary, stream.read()


class _TestCookieHeaders:
    """A headers adapter for cookielib"""

    def __init__(self, headers: t.Union[Headers, t.List[t.Tuple[str, str]]]) -> None:
        self.headers = headers

    def getheaders(self, name: str) -> t.Iterable[str]:
        headers = []
        name = name.lower()
        for k, v in self.headers:
            if k.lower() == name:
                headers.append(v)
        return headers

    def get_all(
        self, name: str, default: t.Optional[t.Iterable[str]] = None
    ) -> t.Iterable[str]:
        headers = self.getheaders(name)

        if not headers:
            return default  # type: ignore

        return headers


class _TestCookieResponse:
    """Something that looks like a httplib.HTTPResponse, but is actually just an
    adapter for our test responses to make them available for cookielib.
    """

    def __init__(self, headers: t.Union[Headers, t.List[t.Tuple[str, str]]]) -> None:
        self.headers = _TestCookieHeaders(headers)

    def info(self) -> _TestCookieHeaders:
        return self.headers


class _TestCookieJar(CookieJar):
    """A cookielib.CookieJar modified to inject and read cookie headers from
    and to wsgi environments, and wsgi application responses.
    """

    def inject_wsgi(self, environ: "WSGIEnvironment") -> None:
        """Inject the cookies as client headers into the server's wsgi
        environment.
        """
        cvals = [f"{c.name}={c.value}" for c in self]

        if cvals:
            environ["HTTP_COOKIE"] = "; ".join(cvals)
        else:
            environ.pop("HTTP_COOKIE", None)

    def extract_wsgi(
        self,
        environ: "WSGIEnvironment",
        headers: t.Union[Headers, t.List[t.Tuple[str, str]]],
    ) -> None:
        """Extract the server's set-cookie headers as cookies into the
        cookie jar.
        """
        self.extract_cookies(
            _TestCookieResponse(headers),  # type: ignore
            _UrllibRequest(get_current_url(environ)),
        )


def _iter_data(data: t.Mapping[str, t.Any]) -> t.Iterator[t.Tuple[str, t.Any]]:
    """Iterate over a mapping that might have a list of values, yielding
    all key, value pairs. Almost like iter_multi_items but only allows
    lists, not tuples, of values so tuples can be used for files.
    """
    if isinstance(data, MultiDict):
        yield from data.items(multi=True)
    else:
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    yield key, v
            else:
                yield key, value


_TAnyMultiDict = t.TypeVar("_TAnyMultiDict", bound=MultiDict)


class EnvironBuilder:
    """This class can be used to conveniently create a WSGI environment
    for testing purposes.  It can be used to quickly create WSGI environments
    or request objects from arbitrary data.

    The signature of this class is also used in some other places as of
    Werkzeug 0.5 (:func:`create_environ`, :meth:`Response.from_values`,
    :meth:`Client.open`).  Because of this most of the functionality is
    available through the constructor alone.

    Files and regular form data can be manipulated independently of each
    other with the :attr:`form` and :attr:`files` attributes, but are
    passed with the same argument to the constructor: `data`.

    `data` can be any of these values:

    -   a `str` or `bytes` object: The object is converted into an
        :attr:`input_stream`, the :attr:`content_length` is set and you have to
        provide a :attr:`content_type`.
    -   a `dict` or :class:`MultiDict`: The keys have to be strings. The values
        have to be either any of the following objects, or a list of any of the
        following objects:

        -   a :class:`file`-like object:  These are converted into
            :class:`FileStorage` objects automatically.
        -   a `tuple`:  The :meth:`~FileMultiDict.add_file` method is called
            with the key and the unpacked `tuple` items as positional
            arguments.
        -   a `str`:  The string is set as form data for the associated key.
    -   a file-like object: The object content is loaded in memory and then
        handled like a regular `str` or a `bytes`.

    :param path: the path of the request.  In the WSGI environment this will
                 end up as `PATH_INFO`.  If the `query_string` is not defined
                 and there is a question mark in the `path` everything after
                 it is used as query string.
    :param base_url: the base URL is a URL that is used to extract the WSGI
                     URL scheme, host (server name + server port) and the
                     script root (`SCRIPT_NAME`).
    :param query_string: an optional string or dict with URL parameters.
    :param method: the HTTP method to use, defaults to `GET`.
    :param input_stream: an optional input stream.  Do not specify this and
                         `data`.  As soon as an input stream is set you can't
                         modify :attr:`args` and :attr:`files` unless you
                         set the :attr:`input_stream` to `None` again.
    :param content_type: The content type for the request.  As of 0.5 you
                         don't have to provide this when specifying files
                         and form data via `data`.
    :param content_length: The content length for the request.  You don't
                           have to specify this when providing data via
                           `data`.
    :param errors_stream: an optional error stream that is used for
                          `wsgi.errors`.  Defaults to :data:`stderr`.
    :param multithread: controls `wsgi.multithread`.  Defaults to `False`.
    :param multiprocess: controls `wsgi.multiprocess`.  Defaults to `False`.
    :param run_once: controls `wsgi.run_once`.  Defaults to `False`.
    :param headers: an optional list or :class:`Headers` object of headers.
    :param data: a string or dict of form data or a file-object.
                 See explanation above.
    :param json: An object to be serialized and assigned to ``data``.
        Defaults the content type to ``"application/json"``.
        Serialized with the function assigned to :attr:`json_dumps`.
    :param environ_base: an optional dict of environment defaults.
    :param environ_overrides: an optional dict of environment overrides.
    :param charset: the charset used to encode string data.
    :param auth: An authorization object to use for the
        ``Authorization`` header value. A ``(username, password)`` tuple
        is a shortcut for ``Basic`` authorization.

    .. versionchanged:: 2.0
        ``REQUEST_URI`` and ``RAW_URI`` is the full raw URI including
        the query string, not only the path.

    .. versionchanged:: 2.0
        The default :attr:`request_class` is ``Request`` instead of
        ``BaseRequest``.

    .. versionadded:: 2.0
       Added the ``auth`` parameter.

    .. versionadded:: 0.15
        The ``json`` param and :meth:`json_dumps` method.

    .. versionadded:: 0.15
        The environ has keys ``REQUEST_URI`` and ``RAW_URI`` containing
        the path before perecent-decoding. This is not part of the WSGI
        PEP, but many WSGI servers include it.

    .. versionchanged:: 0.6
       ``path`` and ``base_url`` can now be unicode strings that are
       encoded with :func:`iri_to_uri`.
    """

    #: the server protocol to use.  defaults to HTTP/1.1
    server_protocol = "HTTP/1.1"

    #: the wsgi version to use.  defaults to (1, 0)
    wsgi_version = (1, 0)

    #: The default request class used by :meth:`get_request`.
    request_class = Request

    import json

    #: The serialization function used when ``json`` is passed.
    json_dumps = staticmethod(json.dumps)
    del json

    _args: t.Optional[MultiDict]
    _query_string: t.Optional[str]
    _input_stream: t.Optional[t.IO[bytes]]
    _form: t.Optional[MultiDict]
    _files: t.Optional[FileMultiDict]

    def __init__(
        self,
        path: str = "/",
        base_url: t.Optional[str] = None,
        query_string: t.Optional[t.Union[t.Mapping[str, str], str]] = None,
        method: str = "GET",
        input_stream: t.Optional[t.IO[bytes]] = None,
        content_type: t.Optional[str] = None,
        content_length: t.Optional[int] = None,
        errors_stream: t.Optional[t.IO[str]] = None,
        multithread: bool = False,
        multiprocess: bool = False,
        run_once: bool = False,
        headers: t.Optional[t.Union[Headers, t.Iterable[t.Tuple[str, str]]]] = None,
        data: t.Optional[
            t.Union[t.IO[bytes], str, bytes, t.Mapping[str, t.Any]]
        ] = None,
        environ_base: t.Optional[t.Mapping[str, t.Any]] = None,
        environ_overrides: t.Optional[t.Mapping[str, t.Any]] = None,
        charset: str = "utf-8",
        mimetype: t.Optional[str] = None,
        json: t.Optional[t.Mapping[str, t.Any]] = None,
        auth: t.Optional[t.Union[Authorization, t.Tuple[str, str]]] = None,
    ) -> None:
        path_s = _make_encode_wrapper(path)
        if query_string is not None and path_s("?") in path:
            raise ValueError("Query string is defined in the path and as an argument")
        request_uri = url_parse(path)
        if query_string is None and path_s("?") in path:
            query_string = request_uri.query
        self.charset = charset
        self.path = iri_to_uri(request_uri.path)
        self.request_uri = path
        if base_url is not None:
            base_url = url_fix(iri_to_uri(base_url, charset), charset)
        self.base_url = base_url  # type: ignore
        if isinstance(query_string, (bytes, str)):
            self.query_string = query_string
        else:
            if query_string is None:
                query_string = MultiDict()
            elif not isinstance(query_string, MultiDict):
                query_string = MultiDict(query_string)
            self.args = query_string
        self.method = method
        if headers is None:
            headers = Headers()
        elif not isinstance(headers, Headers):
            headers = Headers(headers)
        self.headers = headers
        if content_type is not None:
            self.content_type = content_type
        if errors_stream is None:
            errors_stream = sys.stderr
        self.errors_stream = errors_stream
        self.multithread = multithread
        self.multiprocess = multiprocess
        self.run_once = run_once
        self.environ_base = environ_base
        self.environ_overrides = environ_overrides
        self.input_stream = input_stream
        self.content_length = content_length
        self.closed = False

        if auth is not None:
            if isinstance(auth, tuple):
                auth = Authorization(
                    "basic", {"username": auth[0], "password": auth[1]}
                )

            self.headers.set("Authorization", auth.to_header())

        if json is not None:
            if data is not None:
                raise TypeError("can't provide both json and data")

            data = self.json_dumps(json)

            if self.content_type is None:
                self.content_type = "application/json"

        if data:
            if input_stream is not None:
                raise TypeError("can't provide input stream and data")
            if hasattr(data, "read"):
                data = data.read()  # type: ignore
            if isinstance(data, str):
                data = data.encode(self.charset)
            if isinstance(data, bytes):
                self.input_stream = BytesIO(data)
                if self.content_length is None:
                    self.content_length = len(data)
            else:
                for key, value in _iter_data(data):  # type: ignore
                    if isinstance(value, (tuple, dict)) or hasattr(value, "read"):
                        self._add_file_from_data(key, value)
                    else:
                        self.form.setlistdefault(key).append(value)

        if mimetype is not None:
            self.mimetype = mimetype

    @classmethod
    def from_environ(
        cls, environ: "WSGIEnvironment", **kwargs: t.Any
    ) -> "EnvironBuilder":
        """Turn an environ dict back into a builder. Any extra kwargs
        override the args extracted from the environ.

        .. versionchanged:: 2.0
            Path and query values are passed through the WSGI decoding
            dance to avoid double encoding.

        .. versionadded:: 0.15
        """
        headers = Headers(EnvironHeaders(environ))
        out = {
            "path": _wsgi_decoding_dance(environ["PATH_INFO"]),
            "base_url": cls._make_base_url(
                environ["wsgi.url_scheme"],
                headers.pop("Host"),
                _wsgi_decoding_dance(environ["SCRIPT_NAME"]),
            ),
            "query_string": _wsgi_decoding_dance(environ["QUERY_STRING"]),
            "method": environ["REQUEST_METHOD"],
            "input_stream": environ["wsgi.input"],
            "content_type": headers.pop("Content-Type", None),
            "content_length": headers.pop("Content-Length", None),
            "errors_stream": environ["wsgi.errors"],
            "multithread": environ["wsgi.multithread"],
            "multiprocess": environ["wsgi.multiprocess"],
            "run_once": environ["wsgi.run_once"],
            "headers": headers,
        }
        out.update(kwargs)
        return cls(**out)

    def _add_file_from_data(
        self,
        key: str,
        value: t.Union[
            t.IO[bytes], t.Tuple[t.IO[bytes], str], t.Tuple[t.IO[bytes], str, str]
        ],
    ) -> None:
        """Called in the EnvironBuilder to add files from the data dict."""
        if isinstance(value, tuple):
            self.files.add_file(key, *value)
        else:
            self.files.add_file(key, value)

    @staticmethod
    def _make_base_url(scheme: str, host: str, script_root: str) -> str:
        return url_unparse((scheme, host, script_root, "", "")).rstrip("/") + "/"

    @property
    def base_url(self) -> str:
        """The base URL is used to extract the URL scheme, host name,
        port, and root path.
        """
        return self._make_base_url(self.url_scheme, self.host, self.script_root)

    @base_url.setter
    def base_url(self, value: t.Optional[str]) -> None:
        if value is None:
            scheme = "http"
            netloc = "localhost"
            script_root = ""
        else:
            scheme, netloc, script_root, qs, anchor = url_parse(value)
            if qs or anchor:
                raise ValueError("base url must not contain a query string or fragment")
        self.script_root = script_root.rstrip("/")
        self.host = netloc
        self.url_scheme = scheme

    @property
    def content_type(self) -> t.Optional[str]:
        """The content type for the request.  Reflected from and to
        the :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        ct = self.headers.get("Content-Type")
        if ct is None and not self._input_stream:
            if self._files:
                return "multipart/form-data"
            if self._form:
                return "application/x-www-form-urlencoded"
            return None
        return ct

    @content_type.setter
    def content_type(self, value: t.Optional[str]) -> None:
        if value is None:
            self.headers.pop("Content-Type", None)
        else:
            self.headers["Content-Type"] = value

    @property
    def mimetype(self) -> t.Optional[str]:
        """The mimetype (content type without charset etc.)

        .. versionadded:: 0.14
        """
        ct = self.content_type
        return ct.split(";")[0].strip() if ct else None

    @mimetype.setter
    def mimetype(self, value: str) -> None:
        self.content_type = get_content_type(value, self.charset)

    @property
    def mimetype_params(self) -> t.Mapping[str, str]:
        """The mimetype parameters as dict.  For example if the
        content type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.14
        """

        def on_update(d: CallbackDict) -> None:
            self.headers["Content-Type"] = dump_options_header(self.mimetype, d)

        d = parse_options_header(self.headers.get("content-type", ""))[1]
        return CallbackDict(d, on_update)

    @property
    def content_length(self) -> t.Optional[int]:
        """The content length as integer.  Reflected from and to the
        :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        return self.headers.get("Content-Length", type=int)

    @content_length.setter
    def content_length(self, value: t.Optional[int]) -> None:
        if value is None:
            self.headers.pop("Content-Length", None)
        else:
            self.headers["Content-Length"] = str(value)

    def _get_form(self, name: str, storage: t.Type[_TAnyMultiDict]) -> _TAnyMultiDict:
        """Common behavior for getting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param storage: Storage class used for the data.
        """
        if self.input_stream is not None:
            raise AttributeError("an input stream is defined")

        rv = getattr(self, name)

        if rv is None:
            rv = storage()
            setattr(self, name, rv)

        return rv  # type: ignore

    def _set_form(self, name: str, value: MultiDict) -> None:
        """Common behavior for setting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param value: Value to assign to the attribute.
        """
        self._input_stream = None
        setattr(self, name, value)

    @property
    def form(self) -> MultiDict:
        """A :class:`MultiDict` of form values."""
        return self._get_form("_form", MultiDict)

    @form.setter
    def form(self, value: MultiDict) -> None:
        self._set_form("_form", value)

    @property
    def files(self) -> FileMultiDict:
        """A :class:`FileMultiDict` of uploaded files. Use
        :meth:`~FileMultiDict.add_file` to add new files.
        """
        return self._get_form("_files", FileMultiDict)

    @files.setter
    def files(self, value: FileMultiDict) -> None:
        self._set_form("_files", value)

    @property
    def input_stream(self) -> t.Optional[t.IO[bytes]]:
        """An optional input stream. This is mutually exclusive with
        setting :attr:`form` and :attr:`files`, setting it will clear
        those. Do not provide this if the method is not ``POST`` or
        another method that has a body.
        """
        return self._input_stream

    @input_stream.setter
    def input_stream(self, value: t.Optional[t.IO[bytes]]) -> None:
        self._input_stream = value
        self._form = None
        self._files = None

    @property
    def query_string(self) -> str:
        """The query string.  If you set this to a string
        :attr:`args` will no longer be available.
        """
        if self._query_string is None:
            if self._args is not None:
                return url_encode(self._args, charset=self.charset)
            return ""
        return self._query_string

    @query_string.setter
    def query_string(self, value: t.Optional[str]) -> None:
        self._query_string = value
        self._args = None

    @property
    def args(self) -> MultiDict:
        """The URL arguments as :class:`MultiDict`."""
        if self._query_string is not None:
            raise AttributeError("a query string is defined")
        if self._args is None:
            self._args = MultiDict()
        return self._args

    @args.setter
    def args(self, value: t.Optional[MultiDict]) -> None:
        self._query_string = None
        self._args = value

    @property
    def server_name(self) -> str:
        """The server name (read-only, use :attr:`host` to set)"""
        return self.host.split(":", 1)[0]

    @property
    def server_port(self) -> int:
        """The server port as integer (read-only, use :attr:`host` to set)"""
        pieces = self.host.split(":", 1)
        if len(pieces) == 2 and pieces[1].isdigit():
            return int(pieces[1])
        if self.url_scheme == "https":
            return 443
        return 80

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Closes all files.  If you put real :class:`file` objects into the
        :attr:`files` dict you can call this method to automatically close
        them all in one go.
        """
        if self.closed:
            return
        try:
            files = self.files.values()
        except AttributeError:
            files = ()  # type: ignore
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        self.closed = True

    def get_environ(self) -> "WSGIEnvironment":
        """Return the built environ.

        .. versionchanged:: 0.15
            The content type and length headers are set based on
            input stream detection. Previously this only set the WSGI
            keys.
        """
        input_stream = self.input_stream
        content_length = self.content_length

        mimetype = self.mimetype
        content_type = self.content_type

        if input_stream is not None:
            start_pos = input_stream.tell()
            input_stream.seek(0, 2)
            end_pos = input_stream.tell()
            input_stream.seek(start_pos)
            content_length = end_pos - start_pos
        elif mimetype == "multipart/form-data":
            input_stream, content_length, boundary = stream_encode_multipart(
                CombinedMultiDict([self.form, self.files]), charset=self.charset
            )
            content_type = f'{mimetype}; boundary="{boundary}"'
        elif mimetype == "application/x-www-form-urlencoded":
            form_encoded = url_encode(self.form, charset=self.charset).encode("ascii")
            content_length = len(form_encoded)
            input_stream = BytesIO(form_encoded)
        else:
            input_stream = BytesIO()

        result: "WSGIEnvironment" = {}
        if self.environ_base:
            result.update(self.environ_base)

        def _path_encode(x: str) -> str:
            return _wsgi_encoding_dance(url_unquote(x, self.charset), self.charset)

        raw_uri = _wsgi_encoding_dance(self.request_uri, self.charset)
        result.update(
            {
                "REQUEST_METHOD": self.method,
                "SCRIPT_NAME": _path_encode(self.script_root),
                "PATH_INFO": _path_encode(self.path),
                "QUERY_STRING": _wsgi_encoding_dance(self.query_string, self.charset),
                # Non-standard, added by mod_wsgi, uWSGI
                "REQUEST_URI": raw_uri,
                # Non-standard, added by gunicorn
                "RAW_URI": raw_uri,
                "SERVER_NAME": self.server_name,
                "SERVER_PORT": str(self.server_port),
                "HTTP_HOST": self.host,
                "SERVER_PROTOCOL": self.server_protocol,
                "wsgi.version": self.wsgi_version,
                "wsgi.url_scheme": self.url_scheme,
                "wsgi.input": input_stream,
                "wsgi.errors": self.errors_stream,
                "wsgi.multithread": self.multithread,
                "wsgi.multiprocess": self.multiprocess,
                "wsgi.run_once": self.run_once,
            }
        )

        headers = self.headers.copy()

        if content_type is not None:
            result["CONTENT_TYPE"] = content_type
            headers.set("Content-Type", content_type)

        if content_length is not None:
            result["CONTENT_LENGTH"] = str(content_length)
            headers.set("Content-Length", content_length)

        combined_headers = defaultdict(list)

        for key, value in headers.to_wsgi_list():
            combined_headers[f"HTTP_{key.upper().replace('-', '_')}"].append(value)

        for key, values in combined_headers.items():
            result[key] = ", ".join(values)

        if self.environ_overrides:
            result.update(self.environ_overrides)

        return result

    def get_request(self, cls: t.Optional[t.Type[Request]] = None) -> Request:
        """Returns a request with the data.  If the request class is not
        specified :attr:`request_class` is used.

        :param cls: The request wrapper to use.
        """
        if cls is None:
            cls = self.request_class

        return cls(self.get_environ())


class ClientRedirectError(Exception):
    """If a redirect loop is detected when using follow_redirects=True with
    the :cls:`Client`, then this exception is raised.
    """


class Client:
    """This class allows you to send requests to a wrapped application.

    The use_cookies parameter indicates whether cookies should be stored and
    sent for subsequent requests. This is True by default, but passing False
    will disable this behaviour.

    If you want to request some subdomain of your application you may set
    `allow_subdomain_redirects` to `True` as if not no external redirects
    are allowed.

    .. versionchanged:: 2.0
        ``response_wrapper`` is always a subclass of
        :class:``TestResponse``.

    .. versionchanged:: 0.5
        Added the ``use_cookies`` parameter.
    """

    def __init__(
        self,
        application: "WSGIApplication",
        response_wrapper: t.Optional[t.Type["Response"]] = None,
        use_cookies: bool = True,
        allow_subdomain_redirects: bool = False,
    ) -> None:
        self.application = application

        if response_wrapper in {None, Response}:
            response_wrapper = TestResponse
        elif not isinstance(response_wrapper, TestResponse):
            response_wrapper = type(
                "WrapperTestResponse",
                (TestResponse, response_wrapper),  # type: ignore
                {},
            )

        self.response_wrapper = t.cast(t.Type["TestResponse"], response_wrapper)

        if use_cookies:
            self.cookie_jar: t.Optional[_TestCookieJar] = _TestCookieJar()
        else:
            self.cookie_jar = None

        self.allow_subdomain_redirects = allow_subdomain_redirects

    def set_cookie(
        self,
        server_name: str,
        key: str,
        value: str = "",
        max_age: t.Optional[t.Union[timedelta, int]] = None,
        expires: t.Optional[t.Union[str, datetime, int, float]] = None,
        path: str = "/",
        domain: t.Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: t.Optional[str] = None,
        charset: str = "utf-8",
    ) -> None:
        """Sets a cookie in the client's cookie jar.  The server name
        is required and has to match the one that is also passed to
        the open call.
        """
        assert self.cookie_jar is not None, "cookies disabled"
        header = dump_cookie(
            key,
            value,
            max_age,
            expires,
            path,
            domain,
            secure,
            httponly,
            charset,
            samesite=samesite,
        )
        environ = create_environ(path, base_url=f"http://{server_name}")
        headers = [("Set-Cookie", header)]
        self.cookie_jar.extract_wsgi(environ, headers)

    def delete_cookie(
        self,
        server_name: str,
        key: str,
        path: str = "/",
        domain: t.Optional[str] = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: t.Optional[str] = None,
    ) -> None:
        """Deletes a cookie in the test client."""
        self.set_cookie(
            server_name,
            key,
            expires=0,
            max_age=0,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )

    def run_wsgi_app(
        self, environ: "WSGIEnvironment", buffered: bool = False
    ) -> t.Tuple[t.Iterable[bytes], str, Headers]:
        """Runs the wrapped WSGI app with the given environment.

        :meta private:
        """
        if self.cookie_jar is not None:
            self.cookie_jar.inject_wsgi(environ)

        rv = run_wsgi_app(self.application, environ, buffered=buffered)

        if self.cookie_jar is not None:
            self.cookie_jar.extract_wsgi(environ, rv[2])

        return rv

    def resolve_redirect(
        self, response: "TestResponse", buffered: bool = False
    ) -> "TestResponse":
        """Perform a new request to the location given by the redirect
        response to the previous request.

        :meta private:
        """
        scheme, netloc, path, qs, anchor = url_parse(response.location)
        builder = EnvironBuilder.from_environ(response.request.environ, query_string=qs)

        to_name_parts = netloc.split(":", 1)[0].split(".")
        from_name_parts = builder.server_name.split(".")

        if to_name_parts != [""]:
            # The new location has a host, use it for the base URL.
            builder.url_scheme = scheme
            builder.host = netloc
        else:
            # A local redirect with autocorrect_location_header=False
            # doesn't have a host, so use the request's host.
            to_name_parts = from_name_parts

        # Explain why a redirect to a different server name won't be followed.
        if to_name_parts != from_name_parts:
            if to_name_parts[-len(from_name_parts) :] == from_name_parts:
                if not self.allow_subdomain_redirects:
                    raise RuntimeError("Following subdomain redirects is not enabled.")
            else:
                raise RuntimeError("Following external redirects is not supported.")

        path_parts = path.split("/")
        root_parts = builder.script_root.split("/")

        if path_parts[: len(root_parts)] == root_parts:
            # Strip the script root from the path.
            builder.path = path[len(builder.script_root) :]
        else:
            # The new location is not under the script root, so use the
            # whole path and clear the previous root.
            builder.path = path
            builder.script_root = ""

        # Only 307 and 308 preserve all of the original request.
        if response.status_code not in {307, 308}:
            # HEAD is preserved, everything else becomes GET.
            if builder.method != "HEAD":
                builder.method = "GET"

            # Clear the body and the headers that describe it.

            if builder.input_stream is not None:
                builder.input_stream.close()
                builder.input_stream = None

            builder.content_type = None
            builder.content_length = None
            builder.headers.pop("Transfer-Encoding", None)

        return self.open(builder, buffered=buffered)

    def open(
        self,
        *args: t.Any,
        as_tuple: bool = False,
        buffered: bool = False,
        follow_redirects: bool = False,
        **kwargs: t.Any,
    ) -> "TestResponse":
        """Generate an environ dict from the given arguments, make a
        request to the application using it, and return the response.

        :param args: Passed to :class:`EnvironBuilder` to create the
            environ for the request. If a single arg is passed, it can
            be an existing :class:`EnvironBuilder` or an environ dict.
        :param buffered: Convert the iterator returned by the app into
            a list. If the iterator has a ``close()`` method, it is
            called automatically.
        :param follow_redirects: Make additional requests to follow HTTP
            redirects until a non-redirect status is returned.
            :attr:`TestResponse.history` lists the intermediate
            responses.

        .. versionchanged:: 2.0
            ``as_tuple`` is deprecated and will be removed in Werkzeug
            2.1. Use :attr:`TestResponse.request` and
            ``request.environ`` instead.

        .. versionchanged:: 2.0
            The request input stream is closed when calling
            ``response.close()``. Input streams for redirects are
            automatically closed.

        .. versionchanged:: 0.5
            If a dict is provided as file in the dict for the ``data``
            parameter the content type has to be called ``content_type``
            instead of ``mimetype``. This change was made for
            consistency with :class:`werkzeug.FileWrapper`.

        .. versionchanged:: 0.5
            Added the ``follow_redirects`` parameter.
        """
        request: t.Optional["Request"] = None

        if not kwargs and len(args) == 1:
            arg = args[0]

            if isinstance(arg, EnvironBuilder):
                request = arg.get_request()
            elif isinstance(arg, dict):
                request = EnvironBuilder.from_environ(arg).get_request()
            elif isinstance(arg, Request):
                request = arg

        if request is None:
            builder = EnvironBuilder(*args, **kwargs)

            try:
                request = builder.get_request()
            finally:
                builder.close()

        response = self.run_wsgi_app(request.environ, buffered=buffered)
        response = self.response_wrapper(*response, request=request)

        redirects = set()
        history: t.List["TestResponse"] = []

        while follow_redirects and response.status_code in {
            301,
            302,
            303,
            305,
            307,
            308,
        }:
            # Exhaust intermediate response bodies to ensure middleware
            # that returns an iterator runs any cleanup code.
            if not buffered:
                response.make_sequence()
                response.close()

            new_redirect_entry = (response.location, response.status_code)

            if new_redirect_entry in redirects:
                raise ClientRedirectError(
                    f"Loop detected: A {response.status_code} redirect"
                    f" to {response.location} was already made."
                )

            redirects.add(new_redirect_entry)
            response.history = tuple(history)
            history.append(response)
            response = self.resolve_redirect(response, buffered=buffered)
        else:
            # This is the final request after redirects, or not
            # following redirects.
            response.history = tuple(history)
            # Close the input stream when closing the response, in case
            # the input is an open temporary file.
            response.call_on_close(request.input_stream.close)

        if as_tuple:
            warnings.warn(
                "'as_tuple' is deprecated and will be removed in"
                " Werkzeug 2.1. Access 'response.request.environ'"
                " instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return request.environ, response  # type: ignore

        return response

    def get(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``GET``."""
        kw["method"] = "GET"
        return self.open(*args, **kw)

    def post(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``POST``."""
        kw["method"] = "POST"
        return self.open(*args, **kw)

    def put(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``PUT``."""
        kw["method"] = "PUT"
        return self.open(*args, **kw)

    def delete(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``DELETE``."""
        kw["method"] = "DELETE"
        return self.open(*args, **kw)

    def patch(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``PATCH``."""
        kw["method"] = "PATCH"
        return self.open(*args, **kw)

    def options(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``OPTIONS``."""
        kw["method"] = "OPTIONS"
        return self.open(*args, **kw)

    def head(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``HEAD``."""
        kw["method"] = "HEAD"
        return self.open(*args, **kw)

    def trace(self, *args: t.Any, **kw: t.Any) -> "TestResponse":
        """Call :meth:`open` with ``method`` set to ``TRACE``."""
        kw["method"] = "TRACE"
        return self.open(*args, **kw)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.application!r}>"


def create_environ(*args: t.Any, **kwargs: t.Any) -> "WSGIEnvironment":
    """Create a new WSGI environ dict based on the values passed.  The first
    parameter should be the path of the request which defaults to '/'.  The
    second one can either be an absolute path (in that case the host is
    localhost:80) or a full path to the request with scheme, netloc port and
    the path to the script.

    This accepts the same arguments as the :class:`EnvironBuilder`
    constructor.

    .. versionchanged:: 0.5
       This function is now a thin wrapper over :class:`EnvironBuilder` which
       was added in 0.5.  The `headers`, `environ_base`, `environ_overrides`
       and `charset` parameters were added.
    """
    builder = EnvironBuilder(*args, **kwargs)

    try:
        return builder.get_environ()
    finally:
        builder.close()


def run_wsgi_app(
    app: "WSGIApplication", environ: "WSGIEnvironment", buffered: bool = False
) -> t.Tuple[t.Iterable[bytes], str, Headers]:
    """Return a tuple in the form (app_iter, status, headers) of the
    application output.  This works best if you pass it an application that
    returns an iterator all the time.

    Sometimes applications may use the `write()` callable returned
    by the `start_response` function.  This tries to resolve such edge
    cases automatically.  But if you don't get the expected output you
    should set `buffered` to `True` which enforces buffering.

    If passed an invalid WSGI application the behavior of this function is
    undefined.  Never pass non-conforming WSGI applications to this function.

    :param app: the application to execute.
    :param buffered: set to `True` to enforce buffering.
    :return: tuple in the form ``(app_iter, status, headers)``
    """
    # Copy environ to ensure any mutations by the app (ProxyFix, for
    # example) don't affect subsequent requests (such as redirects).
    environ = _get_environ(environ).copy()
    status: str
    response: t.Optional[t.Tuple[str, t.List[t.Tuple[str, str]]]] = None
    buffer: t.List[bytes] = []

    def start_response(status, headers, exc_info=None):  # type: ignore
        nonlocal response

        if exc_info:
            try:
                raise exc_info[1].with_traceback(exc_info[2])
            finally:
                exc_info = None

        response = (status, headers)
        return buffer.append

    app_rv = app(environ, start_response)
    close_func = getattr(app_rv, "close", None)
    app_iter: t.Iterable[bytes] = iter(app_rv)

    # when buffering we emit the close call early and convert the
    # application iterator into a regular list
    if buffered:
        try:
            app_iter = list(app_iter)
        finally:
            if close_func is not None:
                close_func()

    # otherwise we iterate the application iter until we have a response, chain
    # the already received data with the already collected data and wrap it in
    # a new `ClosingIterator` if we need to restore a `close` callable from the
    # original return value.
    else:
        for item in app_iter:
            buffer.append(item)

            if response is not None:
                break

        if buffer:
            app_iter = chain(buffer, app_iter)

        if close_func is not None and app_iter is not app_rv:
            app_iter = ClosingIterator(app_iter, close_func)

    status, headers = response  # type: ignore
    return app_iter, status, Headers(headers)


class TestResponse(Response):
    """:class:`~werkzeug.wrappers.Response` subclass that provides extra
    information about requests made with the test :class:`Client`.

    Test client requests will always return an instance of this class.
    If a custom response class is passed to the client, it is
    subclassed along with this to support test information.

    If the test request included large files, or if the application is
    serving a file, call :meth:`close` to close any open files and
    prevent Python showing a ``ResourceWarning``.
    """

    request: Request
    """A request object with the environ used to make the request that
    resulted in this response.
    """

    history: t.Tuple["TestResponse", ...]
    """A list of intermediate responses. Populated when the test request
    is made with ``follow_redirects`` enabled.
    """

    def __init__(
        self,
        response: t.Iterable[bytes],
        status: str,
        headers: Headers,
        request: Request,
        history: t.Tuple["TestResponse"] = (),  # type: ignore
        **kwargs: t.Any,
    ) -> None:
        super().__init__(response, status, headers, **kwargs)
        self.request = request
        self.history = history
        self._compat_tuple = response, status, headers

    def __iter__(self) -> t.Iterator:
        warnings.warn(
            (
                "The test client no longer returns a tuple, it returns"
                " a 'TestResponse'. Tuple unpacking is deprecated and"
                " will be removed in Werkzeug 2.1. Access the"
                " attributes 'data', 'status', and 'headers' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return iter(self._compat_tuple)

    def __getitem__(self, item: int) -> t.Any:
        warnings.warn(
            (
                "The test client no longer returns a tuple, it returns"
                " a 'TestResponse'. Item indexing is deprecated and"
                " will be removed in Werkzeug 2.1. Access the"
                " attributes 'data', 'status', and 'headers' instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
        return self._compat_tuple[item]
