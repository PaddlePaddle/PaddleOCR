import codecs
import io
import mimetypes
import os
import pkgutil
import re
import sys
import typing as t
import unicodedata
import warnings
from datetime import datetime
from html.entities import name2codepoint
from time import time
from zlib import adler32

from ._internal import _DictAccessorProperty
from ._internal import _missing
from ._internal import _parse_signature
from ._internal import _TAccessorValue
from .datastructures import Headers
from .exceptions import NotFound
from .exceptions import RequestedRangeNotSatisfiable
from .security import safe_join
from .urls import url_quote
from .wsgi import wrap_file

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment
    from .wrappers.request import Request
    from .wrappers.response import Response

_T = t.TypeVar("_T")

_entity_re = re.compile(r"&([^;]+);")
_filename_ascii_strip_re = re.compile(r"[^A-Za-z0-9_.-]")
_windows_device_files = (
    "CON",
    "AUX",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "LPT1",
    "LPT2",
    "LPT3",
    "PRN",
    "NUL",
)


class cached_property(property, t.Generic[_T]):
    """A :func:`property` that is only evaluated once. Subsequent access
    returns the cached value. Setting the property sets the cached
    value. Deleting the property clears the cached value, accessing it
    again will evaluate it again.

    .. code-block:: python

        class Example:
            @cached_property
            def value(self):
                # calculate something important here
                return 42

        e = Example()
        e.value  # evaluates
        e.value  # uses cache
        e.value = 16  # sets cache
        del e.value  # clears cache

    The class must have a ``__dict__`` for this to work.

    .. versionchanged:: 2.0
        ``del obj.name`` clears the cached value.
    """

    def __init__(
        self,
        fget: t.Callable[[t.Any], _T],
        name: t.Optional[str] = None,
        doc: t.Optional[str] = None,
    ) -> None:
        super().__init__(fget, doc=doc)
        self.__name__ = name or fget.__name__
        self.__module__ = fget.__module__

    def __set__(self, obj: object, value: _T) -> None:
        obj.__dict__[self.__name__] = value

    def __get__(self, obj: object, type: type = None) -> _T:  # type: ignore
        if obj is None:
            return self  # type: ignore

        value: _T = obj.__dict__.get(self.__name__, _missing)

        if value is _missing:
            value = self.fget(obj)  # type: ignore
            obj.__dict__[self.__name__] = value

        return value

    def __delete__(self, obj: object) -> None:
        del obj.__dict__[self.__name__]


def invalidate_cached_property(obj: object, name: str) -> None:
    """Invalidates the cache for a :class:`cached_property`:

    >>> class Test(object):
    ...     @cached_property
    ...     def magic_number(self):
    ...         print("recalculating...")
    ...         return 42
    ...
    >>> var = Test()
    >>> var.magic_number
    recalculating...
    42
    >>> var.magic_number
    42
    >>> invalidate_cached_property(var, "magic_number")
    >>> var.magic_number
    recalculating...
    42

    You must pass the name of the cached property as the second argument.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use ``del obj.name`` instead.
    """
    warnings.warn(
        "'invalidate_cached_property' is deprecated and will be removed"
        " in Werkzeug 2.1. Use 'del obj.name' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    delattr(obj, name)


class environ_property(_DictAccessorProperty[_TAccessorValue]):
    """Maps request attributes to environment variables. This works not only
    for the Werkzeug request object, but also any other class with an
    environ attribute:

    >>> class Test(object):
    ...     environ = {'key': 'value'}
    ...     test = environ_property('key')
    >>> var = Test()
    >>> var.test
    'value'

    If you pass it a second value it's used as default if the key does not
    exist, the third one can be a converter that takes a value and converts
    it.  If it raises :exc:`ValueError` or :exc:`TypeError` the default value
    is used. If no default value is provided `None` is used.

    Per default the property is read only.  You have to explicitly enable it
    by passing ``read_only=False`` to the constructor.
    """

    read_only = True

    def lookup(self, obj: "Request") -> "WSGIEnvironment":
        return obj.environ


class header_property(_DictAccessorProperty[_TAccessorValue]):
    """Like `environ_property` but for headers."""

    def lookup(self, obj: t.Union["Request", "Response"]) -> Headers:
        return obj.headers


class HTMLBuilder:
    """Helper object for HTML generation.

    Per default there are two instances of that class.  The `html` one, and
    the `xhtml` one for those two dialects.  The class uses keyword parameters
    and positional parameters to generate small snippets of HTML.

    Keyword parameters are converted to XML/SGML attributes, positional
    arguments are used as children.  Because Python accepts positional
    arguments before keyword arguments it's a good idea to use a list with the
    star-syntax for some children:

    >>> html.p(class_='foo', *[html.a('foo', href='foo.html'), ' ',
    ...                        html.a('bar', href='bar.html')])
    '<p class="foo"><a href="foo.html">foo</a> <a href="bar.html">bar</a></p>'

    This class works around some browser limitations and can not be used for
    arbitrary SGML/XML generation.  For that purpose lxml and similar
    libraries exist.

    Calling the builder escapes the string passed:

    >>> html.p(html("<foo>"))
    '<p>&lt;foo&gt;</p>'

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1.
    """

    _entity_re = re.compile(r"&([^;]+);")
    _entities = name2codepoint.copy()
    _entities["apos"] = 39
    _empty_elements = {
        "area",
        "base",
        "basefont",
        "br",
        "col",
        "command",
        "embed",
        "frame",
        "hr",
        "img",
        "input",
        "keygen",
        "isindex",
        "link",
        "meta",
        "param",
        "source",
        "wbr",
    }
    _boolean_attributes = {
        "selected",
        "checked",
        "compact",
        "declare",
        "defer",
        "disabled",
        "ismap",
        "multiple",
        "nohref",
        "noresize",
        "noshade",
        "nowrap",
    }
    _plaintext_elements = {"textarea"}
    _c_like_cdata = {"script", "style"}

    def __init__(self, dialect):  # type: ignore
        self._dialect = dialect

    def __call__(self, s):  # type: ignore
        import html

        warnings.warn(
            "'utils.HTMLBuilder' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
        return html.escape(s)

    def __getattr__(self, tag):  # type: ignore
        import html

        warnings.warn(
            "'utils.HTMLBuilder' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
        if tag[:2] == "__":
            raise AttributeError(tag)

        def proxy(*children, **arguments):  # type: ignore
            buffer = f"<{tag}"
            for key, value in arguments.items():
                if value is None:
                    continue
                if key[-1] == "_":
                    key = key[:-1]
                if key in self._boolean_attributes:
                    if not value:
                        continue
                    if self._dialect == "xhtml":
                        value = f'="{key}"'
                    else:
                        value = ""
                else:
                    value = f'="{html.escape(value)}"'
                buffer += f" {key}{value}"
            if not children and tag in self._empty_elements:
                if self._dialect == "xhtml":
                    buffer += " />"
                else:
                    buffer += ">"
                return buffer
            buffer += ">"

            children_as_string = "".join([str(x) for x in children if x is not None])

            if children_as_string:
                if tag in self._plaintext_elements:
                    children_as_string = html.escape(children_as_string)
                elif tag in self._c_like_cdata and self._dialect == "xhtml":
                    children_as_string = f"/*<![CDATA[*/{children_as_string}/*]]>*/"
            buffer += children_as_string + f"</{tag}>"
            return buffer

        return proxy

    def __repr__(self) -> str:
        return f"<{type(self).__name__} for {self._dialect!r}>"


html = HTMLBuilder("html")
xhtml = HTMLBuilder("xhtml")

# https://cgit.freedesktop.org/xdg/shared-mime-info/tree/freedesktop.org.xml.in
# https://www.iana.org/assignments/media-types/media-types.xhtml
# Types listed in the XDG mime info that have a charset in the IANA registration.
_charset_mimetypes = {
    "application/ecmascript",
    "application/javascript",
    "application/sql",
    "application/xml",
    "application/xml-dtd",
    "application/xml-external-parsed-entity",
}


def get_content_type(mimetype: str, charset: str) -> str:
    """Returns the full content type string with charset for a mimetype.

    If the mimetype represents text, the charset parameter will be
    appended, otherwise the mimetype is returned unchanged.

    :param mimetype: The mimetype to be used as content type.
    :param charset: The charset to be appended for text mimetypes.
    :return: The content type.

    .. versionchanged:: 0.15
        Any type that ends with ``+xml`` gets a charset, not just those
        that start with ``application/``. Known text types such as
        ``application/javascript`` are also given charsets.
    """
    if (
        mimetype.startswith("text/")
        or mimetype in _charset_mimetypes
        or mimetype.endswith("+xml")
    ):
        mimetype += f"; charset={charset}"

    return mimetype


def detect_utf_encoding(data: bytes) -> str:
    """Detect which UTF encoding was used to encode the given bytes.

    The latest JSON standard (:rfc:`8259`) suggests that only UTF-8 is
    accepted. Older documents allowed 8, 16, or 32. 16 and 32 can be big
    or little endian. Some editors or libraries may prepend a BOM.

    :internal:

    :param data: Bytes in unknown UTF encoding.
    :return: UTF encoding name

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. This is built in to
        :func:`json.loads`.

    .. versionadded:: 0.15
    """
    warnings.warn(
        "'detect_utf_encoding' is deprecated and will be removed in"
        " Werkzeug 2.1. This is built in to 'json.loads'.",
        DeprecationWarning,
        stacklevel=2,
    )
    head = data[:4]

    if head[:3] == codecs.BOM_UTF8:
        return "utf-8-sig"

    if b"\x00" not in head:
        return "utf-8"

    if head in (codecs.BOM_UTF32_BE, codecs.BOM_UTF32_LE):
        return "utf-32"

    if head[:2] in (codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE):
        return "utf-16"

    if len(head) == 4:
        if head[:3] == b"\x00\x00\x00":
            return "utf-32-be"

        if head[::2] == b"\x00\x00":
            return "utf-16-be"

        if head[1:] == b"\x00\x00\x00":
            return "utf-32-le"

        if head[1::2] == b"\x00\x00":
            return "utf-16-le"

    if len(head) == 2:
        return "utf-16-be" if head.startswith(b"\x00") else "utf-16-le"

    return "utf-8"


def format_string(string: str, context: t.Mapping[str, t.Any]) -> str:
    """String-template format a string:

    >>> format_string('$foo and ${foo}s', dict(foo=42))
    '42 and 42s'

    This does not do any attribute lookup.

    :param string: the format string.
    :param context: a dict with the variables to insert.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use :class:`string.Template`
        instead.
    """
    from string import Template

    warnings.warn(
        "'utils.format_string' is deprecated and will be removed in"
        " Werkzeug 2.1. Use 'string.Template' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return Template(string).substitute(context)


def secure_filename(filename: str) -> str:
    r"""Pass it a filename and it will return a secure version of it.  This
    filename can then safely be stored on a regular file system and passed
    to :func:`os.path.join`.  The filename returned is an ASCII only string
    for maximum portability.

    On windows systems the function also makes sure that the file is not
    named after one of the special device files.

    >>> secure_filename("My cool movie.mov")
    'My_cool_movie.mov'
    >>> secure_filename("../../../etc/passwd")
    'etc_passwd'
    >>> secure_filename('i contain cool \xfcml\xe4uts.txt')
    'i_contain_cool_umlauts.txt'

    The function might return an empty filename.  It's your responsibility
    to ensure that the filename is unique and that you abort or
    generate a random filename if the function returned an empty one.

    .. versionadded:: 0.5

    :param filename: the filename to secure
    """
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")

    for sep in os.path.sep, os.path.altsep:
        if sep:
            filename = filename.replace(sep, " ")
    filename = str(_filename_ascii_strip_re.sub("", "_".join(filename.split()))).strip(
        "._"
    )

    # on nt a couple of special files are present in each folder.  We
    # have to ensure that the target file is not such a filename.  In
    # this case we prepend an underline
    if (
        os.name == "nt"
        and filename
        and filename.split(".")[0].upper() in _windows_device_files
    ):
        filename = f"_{filename}"

    return filename


def escape(s: t.Any) -> str:
    """Replace ``&``, ``<``, ``>``, ``"``, and ``'`` with HTML-safe
    sequences.

    ``None`` is escaped to an empty string.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use MarkupSafe instead.
    """
    import html

    warnings.warn(
        "'utils.escape' is deprecated and will be removed in Werkzeug"
        " 2.1. Use MarkupSafe instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if s is None:
        return ""

    if hasattr(s, "__html__"):
        return s.__html__()  # type: ignore

    if not isinstance(s, str):
        s = str(s)

    return html.escape(s, quote=True)  # type: ignore


def unescape(s: str) -> str:
    """The reverse of :func:`escape`. This unescapes all the HTML
    entities, not only those inserted by ``escape``.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use MarkupSafe instead.
    """
    import html

    warnings.warn(
        "'utils.unescape' is deprecated and will be removed in Werkzueg"
        " 2.1. Use MarkupSafe instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return html.unescape(s)


def redirect(
    location: str, code: int = 302, Response: t.Optional[t.Type["Response"]] = None
) -> "Response":
    """Returns a response object (a WSGI application) that, if called,
    redirects the client to the target location. Supported codes are
    301, 302, 303, 305, 307, and 308. 300 is not supported because
    it's not a real redirect and 304 because it's the answer for a
    request with a request with defined If-Modified-Since headers.

    .. versionadded:: 0.6
       The location can now be a unicode string that is encoded using
       the :func:`iri_to_uri` function.

    .. versionadded:: 0.10
        The class used for the Response object can now be passed in.

    :param location: the location the response should redirect to.
    :param code: the redirect status code. defaults to 302.
    :param class Response: a Response class to use when instantiating a
        response. The default is :class:`werkzeug.wrappers.Response` if
        unspecified.
    """
    import html

    if Response is None:
        from .wrappers import Response  # type: ignore

    display_location = html.escape(location)
    if isinstance(location, str):
        # Safe conversion is necessary here as we might redirect
        # to a broken URI scheme (for instance itms-services).
        from .urls import iri_to_uri

        location = iri_to_uri(location, safe_conversion=True)
    response = Response(  # type: ignore
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2 Final//EN">\n'
        "<title>Redirecting...</title>\n"
        "<h1>Redirecting...</h1>\n"
        "<p>You should be redirected automatically to target URL: "
        f'<a href="{html.escape(location)}">{display_location}</a>. If'
        " not click the link.",
        code,
        mimetype="text/html",
    )
    response.headers["Location"] = location
    return response


def append_slash_redirect(environ: "WSGIEnvironment", code: int = 301) -> "Response":
    """Redirects to the same URL but with a slash appended.  The behavior
    of this function is undefined if the path ends with a slash already.

    :param environ: the WSGI environment for the request that triggers
                    the redirect.
    :param code: the status code for the redirect.
    """
    new_path = environ["PATH_INFO"].strip("/") + "/"
    query_string = environ.get("QUERY_STRING")
    if query_string:
        new_path += f"?{query_string}"
    return redirect(new_path, code)


def send_file(
    path_or_file: t.Union[os.PathLike, str, t.IO[bytes]],
    environ: "WSGIEnvironment",
    mimetype: t.Optional[str] = None,
    as_attachment: bool = False,
    download_name: t.Optional[str] = None,
    conditional: bool = True,
    etag: t.Union[bool, str] = True,
    last_modified: t.Optional[t.Union[datetime, int, float]] = None,
    max_age: t.Optional[
        t.Union[int, t.Callable[[t.Optional[str]], t.Optional[int]]]
    ] = None,
    use_x_sendfile: bool = False,
    response_class: t.Optional[t.Type["Response"]] = None,
    _root_path: t.Optional[t.Union[os.PathLike, str]] = None,
) -> "Response":
    """Send the contents of a file to the client.

    The first argument can be a file path or a file-like object. Paths
    are preferred in most cases because Werkzeug can manage the file and
    get extra information from the path. Passing a file-like object
    requires that the file is opened in binary mode, and is mostly
    useful when building a file in memory with :class:`io.BytesIO`.

    Never pass file paths provided by a user. The path is assumed to be
    trusted, so a user could craft a path to access a file you didn't
    intend.

    If the WSGI server sets a ``file_wrapper`` in ``environ``, it is
    used, otherwise Werkzeug's built-in wrapper is used. Alternatively,
    if the HTTP server supports ``X-Sendfile``, ``use_x_sendfile=True``
    will tell the server to send the given path, which is much more
    efficient than reading it in Python.

    :param path_or_file: The path to the file to send, relative to the
        current working directory if a relative path is given.
        Alternatively, a file-like object opened in binary mode. Make
        sure the file pointer is seeked to the start of the data.
    :param environ: The WSGI environ for the current request.
    :param mimetype: The MIME type to send for the file. If not
        provided, it will try to detect it from the file name.
    :param as_attachment: Indicate to a browser that it should offer to
        save the file instead of displaying it.
    :param download_name: The default name browsers will use when saving
        the file. Defaults to the passed file name.
    :param conditional: Enable conditional and range responses based on
        request headers. Requires passing a file path and ``environ``.
    :param etag: Calculate an ETag for the file, which requires passing
        a file path. Can also be a string to use instead.
    :param last_modified: The last modified time to send for the file,
        in seconds. If not provided, it will try to detect it from the
        file path.
    :param max_age: How long the client should cache the file, in
        seconds. If set, ``Cache-Control`` will be ``public``, otherwise
        it will be ``no-cache`` to prefer conditional caching.
    :param use_x_sendfile: Set the ``X-Sendfile`` header to let the
        server to efficiently send the file. Requires support from the
        HTTP server. Requires passing a file path.
    :param response_class: Build the response using this class. Defaults
        to :class:`~werkzeug.wrappers.Response`.
    :param _root_path: Do not use. For internal use only. Use
        :func:`send_from_directory` to safely send files under a path.

    .. versionchanged:: 2.0.2
        ``send_file`` only sets a detected ``Content-Encoding`` if
        ``as_attachment`` is disabled.

    .. versionadded:: 2.0
        Adapted from Flask's implementation.

    .. versionchanged:: 2.0
        ``download_name`` replaces Flask's ``attachment_filename``
         parameter. If ``as_attachment=False``, it is passed with
         ``Content-Disposition: inline`` instead.

    .. versionchanged:: 2.0
        ``max_age`` replaces Flask's ``cache_timeout`` parameter.
        ``conditional`` is enabled and ``max_age`` is not set by
        default.

    .. versionchanged:: 2.0
        ``etag`` replaces Flask's ``add_etags`` parameter. It can be a
        string to use instead of generating one.

    .. versionchanged:: 2.0
        If an encoding is returned when guessing ``mimetype`` from
        ``download_name``, set the ``Content-Encoding`` header.
    """
    if response_class is None:
        from .wrappers import Response

        response_class = Response

    path: t.Optional[str] = None
    file: t.Optional[t.IO[bytes]] = None
    size: t.Optional[int] = None
    mtime: t.Optional[float] = None
    headers = Headers()

    if isinstance(path_or_file, (os.PathLike, str)) or hasattr(
        path_or_file, "__fspath__"
    ):
        path_or_file = t.cast(t.Union[os.PathLike, str], path_or_file)

        # Flask will pass app.root_path, allowing its send_file wrapper
        # to not have to deal with paths.
        if _root_path is not None:
            path = os.path.join(_root_path, path_or_file)
        else:
            path = os.path.abspath(path_or_file)

        stat = os.stat(path)
        size = stat.st_size
        mtime = stat.st_mtime
    else:
        file = path_or_file

    if download_name is None and path is not None:
        download_name = os.path.basename(path)

    if mimetype is None:
        if download_name is None:
            raise TypeError(
                "Unable to detect the MIME type because a file name is"
                " not available. Either set 'download_name', pass a"
                " path instead of a file, or set 'mimetype'."
            )

        mimetype, encoding = mimetypes.guess_type(download_name)

        if mimetype is None:
            mimetype = "application/octet-stream"

        # Don't send encoding for attachments, it causes browsers to
        # save decompress tar.gz files.
        if encoding is not None and not as_attachment:
            headers.set("Content-Encoding", encoding)

    if download_name is not None:
        try:
            download_name.encode("ascii")
        except UnicodeEncodeError:
            simple = unicodedata.normalize("NFKD", download_name)
            simple = simple.encode("ascii", "ignore").decode("ascii")
            quoted = url_quote(download_name, safe="")
            names = {"filename": simple, "filename*": f"UTF-8''{quoted}"}
        else:
            names = {"filename": download_name}

        value = "attachment" if as_attachment else "inline"
        headers.set("Content-Disposition", value, **names)
    elif as_attachment:
        raise TypeError(
            "No name provided for attachment. Either set"
            " 'download_name' or pass a path instead of a file."
        )

    if use_x_sendfile and path is not None:
        headers["X-Sendfile"] = path
        data = None
    else:
        if file is None:
            file = open(path, "rb")  # type: ignore
        elif isinstance(file, io.BytesIO):
            size = file.getbuffer().nbytes
        elif isinstance(file, io.TextIOBase):
            raise ValueError("Files must be opened in binary mode or use BytesIO.")

        data = wrap_file(environ, file)

    rv = response_class(
        data, mimetype=mimetype, headers=headers, direct_passthrough=True
    )

    if size is not None:
        rv.content_length = size

    if last_modified is not None:
        rv.last_modified = last_modified  # type: ignore
    elif mtime is not None:
        rv.last_modified = mtime  # type: ignore

    rv.cache_control.no_cache = True

    # Flask will pass app.get_send_file_max_age, allowing its send_file
    # wrapper to not have to deal with paths.
    if callable(max_age):
        max_age = max_age(path)

    if max_age is not None:
        if max_age > 0:
            rv.cache_control.no_cache = None
            rv.cache_control.public = True

        rv.cache_control.max_age = max_age
        rv.expires = int(time() + max_age)  # type: ignore

    if isinstance(etag, str):
        rv.set_etag(etag)
    elif etag and path is not None:
        check = adler32(path.encode("utf-8")) & 0xFFFFFFFF
        rv.set_etag(f"{mtime}-{size}-{check}")

    if conditional:
        try:
            rv = rv.make_conditional(environ, accept_ranges=True, complete_length=size)
        except RequestedRangeNotSatisfiable:
            if file is not None:
                file.close()

            raise

        # Some x-sendfile implementations incorrectly ignore the 304
        # status code and send the file anyway.
        if rv.status_code == 304:
            rv.headers.pop("x-sendfile", None)

    return rv


def send_from_directory(
    directory: t.Union[os.PathLike, str],
    path: t.Union[os.PathLike, str],
    environ: "WSGIEnvironment",
    **kwargs: t.Any,
) -> "Response":
    """Send a file from within a directory using :func:`send_file`.

    This is a secure way to serve files from a folder, such as static
    files or uploads. Uses :func:`~werkzeug.security.safe_join` to
    ensure the path coming from the client is not maliciously crafted to
    point outside the specified directory.

    If the final path does not point to an existing regular file,
    returns a 404 :exc:`~werkzeug.exceptions.NotFound` error.

    :param directory: The directory that ``path`` must be located under.
    :param path: The path to the file to send, relative to
        ``directory``.
    :param environ: The WSGI environ for the current request.
    :param kwargs: Arguments to pass to :func:`send_file`.

    .. versionadded:: 2.0
        Adapted from Flask's implementation.
    """
    path = safe_join(os.fspath(directory), os.fspath(path))

    if path is None:
        raise NotFound()

    # Flask will pass app.root_path, allowing its send_from_directory
    # wrapper to not have to deal with paths.
    if "_root_path" in kwargs:
        path = os.path.join(kwargs["_root_path"], path)

    try:
        if not os.path.isfile(path):
            raise NotFound()
    except ValueError:
        # path contains null byte on Python < 3.8
        raise NotFound() from None

    return send_file(path, environ, **kwargs)


def import_string(import_name: str, silent: bool = False) -> t.Any:
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If `silent` is True the return value will be `None` if the import fails.

    :param import_name: the dotted name for the object to import.
    :param silent: if set to `True` import errors are ignored and
                   `None` is returned instead.
    :return: imported object
    """
    import_name = import_name.replace(":", ".")
    try:
        try:
            __import__(import_name)
        except ImportError:
            if "." not in import_name:
                raise
        else:
            return sys.modules[import_name]

        module_name, obj_name = import_name.rsplit(".", 1)
        module = __import__(module_name, globals(), locals(), [obj_name])
        try:
            return getattr(module, obj_name)
        except AttributeError as e:
            raise ImportError(e) from None

    except ImportError as e:
        if not silent:
            raise ImportStringError(import_name, e).with_traceback(
                sys.exc_info()[2]
            ) from None

    return None


def find_modules(
    import_path: str, include_packages: bool = False, recursive: bool = False
) -> t.Iterator[str]:
    """Finds all the modules below a package.  This can be useful to
    automatically import all views / controllers so that their metaclasses /
    function decorators have a chance to register themselves on the
    application.

    Packages are not returned unless `include_packages` is `True`.  This can
    also recursively list modules but in that case it will import all the
    packages to get the correct load path of that module.

    :param import_path: the dotted name for the package to find child modules.
    :param include_packages: set to `True` if packages should be returned, too.
    :param recursive: set to `True` if recursion should happen.
    :return: generator
    """
    module = import_string(import_path)
    path = getattr(module, "__path__", None)
    if path is None:
        raise ValueError(f"{import_path!r} is not a package")
    basename = f"{module.__name__}."
    for _importer, modname, ispkg in pkgutil.iter_modules(path):
        modname = basename + modname
        if ispkg:
            if include_packages:
                yield modname
            if recursive:
                yield from find_modules(modname, include_packages, True)
        else:
            yield modname


def validate_arguments(func, args, kwargs, drop_extra=True):  # type: ignore
    """Checks if the function accepts the arguments and keyword arguments.
    Returns a new ``(args, kwargs)`` tuple that can safely be passed to
    the function without causing a `TypeError` because the function signature
    is incompatible.  If `drop_extra` is set to `True` (which is the default)
    any extra positional or keyword arguments are dropped automatically.

    The exception raised provides three attributes:

    `missing`
        A set of argument names that the function expected but where
        missing.

    `extra`
        A dict of keyword arguments that the function can not handle but
        where provided.

    `extra_positional`
        A list of values that where given by positional argument but the
        function cannot accept.

    This can be useful for decorators that forward user submitted data to
    a view function::

        from werkzeug.utils import ArgumentValidationError, validate_arguments

        def sanitize(f):
            def proxy(request):
                data = request.values.to_dict()
                try:
                    args, kwargs = validate_arguments(f, (request,), data)
                except ArgumentValidationError:
                    raise BadRequest('The browser failed to transmit all '
                                     'the data expected.')
                return f(*args, **kwargs)
            return proxy

    :param func: the function the validation is performed against.
    :param args: a tuple of positional arguments.
    :param kwargs: a dict of keyword arguments.
    :param drop_extra: set to `False` if you don't want extra arguments
                       to be silently dropped.
    :return: tuple in the form ``(args, kwargs)``.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use :func:`inspect.signature`
        instead.
    """
    warnings.warn(
        "'utils.validate_arguments' is deprecated and will be removed"
        " in Werkzeug 2.1. Use 'inspect.signature' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = _parse_signature(func)
    args, kwargs, missing, extra, extra_positional = parser(args, kwargs)[:5]
    if missing:
        raise ArgumentValidationError(tuple(missing))
    elif (extra or extra_positional) and not drop_extra:
        raise ArgumentValidationError(None, extra, extra_positional)
    return tuple(args), kwargs


def bind_arguments(func, args, kwargs):  # type: ignore
    """Bind the arguments provided into a dict.  When passed a function,
    a tuple of arguments and a dict of keyword arguments `bind_arguments`
    returns a dict of names as the function would see it.  This can be useful
    to implement a cache decorator that uses the function arguments to build
    the cache key based on the values of the arguments.

    :param func: the function the arguments should be bound for.
    :param args: tuple of positional arguments.
    :param kwargs: a dict of keyword arguments.
    :return: a :class:`dict` of bound keyword arguments.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use :meth:`Signature.bind`
        instead.
    """
    warnings.warn(
        "'utils.bind_arguments' is deprecated and will be removed in"
        " Werkzeug 2.1. Use 'Signature.bind' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    (
        args,
        kwargs,
        missing,
        extra,
        extra_positional,
        arg_spec,
        vararg_var,
        kwarg_var,
    ) = _parse_signature(func)(args, kwargs)
    values = {}
    for (name, _has_default, _default), value in zip(arg_spec, args):
        values[name] = value
    if vararg_var is not None:
        values[vararg_var] = tuple(extra_positional)
    elif extra_positional:
        raise TypeError("too many positional arguments")
    if kwarg_var is not None:
        multikw = set(extra) & {x[0] for x in arg_spec}
        if multikw:
            raise TypeError(
                f"got multiple values for keyword argument {next(iter(multikw))!r}"
            )
        values[kwarg_var] = extra
    elif extra:
        raise TypeError(f"got unexpected keyword argument {next(iter(extra))!r}")
    return values


class ArgumentValidationError(ValueError):
    """Raised if :func:`validate_arguments` fails to validate

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1 along with ``utils.bind`` and
        ``validate_arguments``.
    """

    def __init__(self, missing=None, extra=None, extra_positional=None):  # type: ignore
        self.missing = set(missing or ())
        self.extra = extra or {}
        self.extra_positional = extra_positional or []
        super().__init__(
            "function arguments invalid."
            f" ({len(self.missing)} missing,"
            f" {len(self.extra) + len(self.extra_positional)} additional)"
        )


class ImportStringError(ImportError):
    """Provides information about a failed :func:`import_string` attempt."""

    #: String in dotted notation that failed to be imported.
    import_name: str
    #: Wrapped exception.
    exception: BaseException

    def __init__(self, import_name: str, exception: BaseException) -> None:
        self.import_name = import_name
        self.exception = exception
        msg = import_name
        name = ""
        tracked = []
        for part in import_name.replace(":", ".").split("."):
            name = f"{name}.{part}" if name else part
            imported = import_string(name, silent=True)
            if imported:
                tracked.append((name, getattr(imported, "__file__", None)))
            else:
                track = [f"- {n!r} found in {i!r}." for n, i in tracked]
                track.append(f"- {name!r} not found.")
                track_str = "\n".join(track)
                msg = (
                    f"import_string() failed for {import_name!r}. Possible reasons"
                    f" are:\n\n"
                    "- missing __init__.py in a package;\n"
                    "- package or module path not included in sys.path;\n"
                    "- duplicated package or module name taking precedence in"
                    " sys.path;\n"
                    "- missing module, class, function or variable;\n\n"
                    f"Debugged import:\n\n{track_str}\n\n"
                    f"Original exception:\n\n{type(exception).__name__}: {exception}"
                )
                break

        super().__init__(msg)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.import_name!r}, {self.exception!r})>"
