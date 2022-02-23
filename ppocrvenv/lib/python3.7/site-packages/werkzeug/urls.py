"""Functions for working with URLs.

Contains implementations of functions from :mod:`urllib.parse` that
handle bytes and strings.
"""
import codecs
import os
import re
import typing as t
import warnings

from ._internal import _check_str_tuple
from ._internal import _decode_idna
from ._internal import _encode_idna
from ._internal import _make_encode_wrapper
from ._internal import _to_str

if t.TYPE_CHECKING:
    from . import datastructures as ds

# A regular expression for what a valid schema looks like
_scheme_re = re.compile(r"^[a-zA-Z0-9+-.]+$")

# Characters that are safe in any part of an URL.
_always_safe = frozenset(
    bytearray(
        b"abcdefghijklmnopqrstuvwxyz"
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        b"0123456789"
        b"-._~"
    )
)

_hexdigits = "0123456789ABCDEFabcdef"
_hextobyte = {
    f"{a}{b}".encode("ascii"): int(f"{a}{b}", 16)
    for a in _hexdigits
    for b in _hexdigits
}
_bytetohex = [f"%{char:02X}".encode("ascii") for char in range(256)]


class _URLTuple(t.NamedTuple):
    scheme: str
    netloc: str
    path: str
    query: str
    fragment: str


class BaseURL(_URLTuple):
    """Superclass of :py:class:`URL` and :py:class:`BytesURL`."""

    __slots__ = ()
    _at: str
    _colon: str
    _lbracket: str
    _rbracket: str

    def __str__(self) -> str:
        return self.to_url()

    def replace(self, **kwargs: t.Any) -> "BaseURL":
        """Return an URL with the same values, except for those parameters
        given new values by whichever keyword arguments are specified."""
        return self._replace(**kwargs)

    @property
    def host(self) -> t.Optional[str]:
        """The host part of the URL if available, otherwise `None`.  The
        host is either the hostname or the IP address mentioned in the
        URL.  It will not contain the port.
        """
        return self._split_host()[0]

    @property
    def ascii_host(self) -> t.Optional[str]:
        """Works exactly like :attr:`host` but will return a result that
        is restricted to ASCII.  If it finds a netloc that is not ASCII
        it will attempt to idna decode it.  This is useful for socket
        operations when the URL might include internationalized characters.
        """
        rv = self.host
        if rv is not None and isinstance(rv, str):
            try:
                rv = _encode_idna(rv)  # type: ignore
            except UnicodeError:
                rv = rv.encode("ascii", "ignore")  # type: ignore
        return _to_str(rv, "ascii", "ignore")

    @property
    def port(self) -> t.Optional[int]:
        """The port in the URL as an integer if it was present, `None`
        otherwise.  This does not fill in default ports.
        """
        try:
            rv = int(_to_str(self._split_host()[1]))
            if 0 <= rv <= 65535:
                return rv
        except (ValueError, TypeError):
            pass
        return None

    @property
    def auth(self) -> t.Optional[str]:
        """The authentication part in the URL if available, `None`
        otherwise.
        """
        return self._split_netloc()[0]

    @property
    def username(self) -> t.Optional[str]:
        """The username if it was part of the URL, `None` otherwise.
        This undergoes URL decoding and will always be a string.
        """
        rv = self._split_auth()[0]
        if rv is not None:
            return _url_unquote_legacy(rv)
        return None

    @property
    def raw_username(self) -> t.Optional[str]:
        """The username if it was part of the URL, `None` otherwise.
        Unlike :attr:`username` this one is not being decoded.
        """
        return self._split_auth()[0]

    @property
    def password(self) -> t.Optional[str]:
        """The password if it was part of the URL, `None` otherwise.
        This undergoes URL decoding and will always be a string.
        """
        rv = self._split_auth()[1]
        if rv is not None:
            return _url_unquote_legacy(rv)
        return None

    @property
    def raw_password(self) -> t.Optional[str]:
        """The password if it was part of the URL, `None` otherwise.
        Unlike :attr:`password` this one is not being decoded.
        """
        return self._split_auth()[1]

    def decode_query(self, *args: t.Any, **kwargs: t.Any) -> "ds.MultiDict[str, str]":
        """Decodes the query part of the URL.  Ths is a shortcut for
        calling :func:`url_decode` on the query argument.  The arguments and
        keyword arguments are forwarded to :func:`url_decode` unchanged.
        """
        return url_decode(self.query, *args, **kwargs)

    def join(self, *args: t.Any, **kwargs: t.Any) -> "BaseURL":
        """Joins this URL with another one.  This is just a convenience
        function for calling into :meth:`url_join` and then parsing the
        return value again.
        """
        return url_parse(url_join(self, *args, **kwargs))

    def to_url(self) -> str:
        """Returns a URL string or bytes depending on the type of the
        information stored.  This is just a convenience function
        for calling :meth:`url_unparse` for this URL.
        """
        return url_unparse(self)

    def encode_netloc(self) -> str:
        """Encodes the netloc part to an ASCII safe URL as bytes."""
        rv = self.ascii_host or ""
        if ":" in rv:
            rv = f"[{rv}]"
        port = self.port
        if port is not None:
            rv = f"{rv}:{port}"
        auth = ":".join(
            filter(
                None,
                [
                    url_quote(self.raw_username or "", "utf-8", "strict", "/:%"),
                    url_quote(self.raw_password or "", "utf-8", "strict", "/:%"),
                ],
            )
        )
        if auth:
            rv = f"{auth}@{rv}"
        return rv

    def decode_netloc(self) -> str:
        """Decodes the netloc part into a string."""
        rv = _decode_idna(self.host or "")

        if ":" in rv:
            rv = f"[{rv}]"
        port = self.port
        if port is not None:
            rv = f"{rv}:{port}"
        auth = ":".join(
            filter(
                None,
                [
                    _url_unquote_legacy(self.raw_username or "", "/:%@"),
                    _url_unquote_legacy(self.raw_password or "", "/:%@"),
                ],
            )
        )
        if auth:
            rv = f"{auth}@{rv}"
        return rv

    def to_uri_tuple(self) -> "BaseURL":
        """Returns a :class:`BytesURL` tuple that holds a URI.  This will
        encode all the information in the URL properly to ASCII using the
        rules a web browser would follow.

        It's usually more interesting to directly call :meth:`iri_to_uri` which
        will return a string.
        """
        return url_parse(iri_to_uri(self))

    def to_iri_tuple(self) -> "BaseURL":
        """Returns a :class:`URL` tuple that holds a IRI.  This will try
        to decode as much information as possible in the URL without
        losing information similar to how a web browser does it for the
        URL bar.

        It's usually more interesting to directly call :meth:`uri_to_iri` which
        will return a string.
        """
        return url_parse(uri_to_iri(self))

    def get_file_location(
        self, pathformat: t.Optional[str] = None
    ) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        """Returns a tuple with the location of the file in the form
        ``(server, location)``.  If the netloc is empty in the URL or
        points to localhost, it's represented as ``None``.

        The `pathformat` by default is autodetection but needs to be set
        when working with URLs of a specific system.  The supported values
        are ``'windows'`` when working with Windows or DOS paths and
        ``'posix'`` when working with posix paths.

        If the URL does not point to a local file, the server and location
        are both represented as ``None``.

        :param pathformat: The expected format of the path component.
                           Currently ``'windows'`` and ``'posix'`` are
                           supported.  Defaults to ``None`` which is
                           autodetect.
        """
        if self.scheme != "file":
            return None, None

        path = url_unquote(self.path)
        host = self.netloc or None

        if pathformat is None:
            if os.name == "nt":
                pathformat = "windows"
            else:
                pathformat = "posix"

        if pathformat == "windows":
            if path[:1] == "/" and path[1:2].isalpha() and path[2:3] in "|:":
                path = f"{path[1:2]}:{path[3:]}"
            windows_share = path[:3] in ("\\" * 3, "/" * 3)
            import ntpath

            path = ntpath.normpath(path)
            # Windows shared drives are represented as ``\\host\\directory``.
            # That results in a URL like ``file://///host/directory``, and a
            # path like ``///host/directory``. We need to special-case this
            # because the path contains the hostname.
            if windows_share and host is None:
                parts = path.lstrip("\\").split("\\", 1)
                if len(parts) == 2:
                    host, path = parts
                else:
                    host = parts[0]
                    path = ""
        elif pathformat == "posix":
            import posixpath

            path = posixpath.normpath(path)
        else:
            raise TypeError(f"Invalid path format {pathformat!r}")

        if host in ("127.0.0.1", "::1", "localhost"):
            host = None

        return host, path

    def _split_netloc(self) -> t.Tuple[t.Optional[str], str]:
        if self._at in self.netloc:
            auth, _, netloc = self.netloc.partition(self._at)
            return auth, netloc
        return None, self.netloc

    def _split_auth(self) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        auth = self._split_netloc()[0]
        if not auth:
            return None, None
        if self._colon not in auth:
            return auth, None

        username, _, password = auth.partition(self._colon)
        return username, password

    def _split_host(self) -> t.Tuple[t.Optional[str], t.Optional[str]]:
        rv = self._split_netloc()[1]
        if not rv:
            return None, None

        if not rv.startswith(self._lbracket):
            if self._colon in rv:
                host, _, port = rv.partition(self._colon)
                return host, port
            return rv, None

        idx = rv.find(self._rbracket)
        if idx < 0:
            return rv, None

        host = rv[1:idx]
        rest = rv[idx + 1 :]
        if rest.startswith(self._colon):
            return host, rest[1:]
        return host, None


class URL(BaseURL):
    """Represents a parsed URL.  This behaves like a regular tuple but
    also has some extra attributes that give further insight into the
    URL.
    """

    __slots__ = ()
    _at = "@"
    _colon = ":"
    _lbracket = "["
    _rbracket = "]"

    def encode(self, charset: str = "utf-8", errors: str = "replace") -> "BytesURL":
        """Encodes the URL to a tuple made out of bytes.  The charset is
        only being used for the path, query and fragment.
        """
        return BytesURL(
            self.scheme.encode("ascii"),  # type: ignore
            self.encode_netloc(),
            self.path.encode(charset, errors),  # type: ignore
            self.query.encode(charset, errors),  # type: ignore
            self.fragment.encode(charset, errors),  # type: ignore
        )


class BytesURL(BaseURL):
    """Represents a parsed URL in bytes."""

    __slots__ = ()
    _at = b"@"  # type: ignore
    _colon = b":"  # type: ignore
    _lbracket = b"["  # type: ignore
    _rbracket = b"]"  # type: ignore

    def __str__(self) -> str:
        return self.to_url().decode("utf-8", "replace")  # type: ignore

    def encode_netloc(self) -> bytes:  # type: ignore
        """Returns the netloc unchanged as bytes."""
        return self.netloc  # type: ignore

    def decode(self, charset: str = "utf-8", errors: str = "replace") -> "URL":
        """Decodes the URL to a tuple made out of strings.  The charset is
        only being used for the path, query and fragment.
        """
        return URL(
            self.scheme.decode("ascii"),  # type: ignore
            self.decode_netloc(),
            self.path.decode(charset, errors),  # type: ignore
            self.query.decode(charset, errors),  # type: ignore
            self.fragment.decode(charset, errors),  # type: ignore
        )


_unquote_maps: t.Dict[t.FrozenSet[int], t.Dict[bytes, int]] = {frozenset(): _hextobyte}


def _unquote_to_bytes(
    string: t.Union[str, bytes], unsafe: t.Union[str, bytes] = ""
) -> bytes:
    if isinstance(string, str):
        string = string.encode("utf-8")

    if isinstance(unsafe, str):
        unsafe = unsafe.encode("utf-8")

    unsafe = frozenset(bytearray(unsafe))
    groups = iter(string.split(b"%"))
    result = bytearray(next(groups, b""))

    try:
        hex_to_byte = _unquote_maps[unsafe]
    except KeyError:
        hex_to_byte = _unquote_maps[unsafe] = {
            h: b for h, b in _hextobyte.items() if b not in unsafe
        }

    for group in groups:
        code = group[:2]

        if code in hex_to_byte:
            result.append(hex_to_byte[code])
            result.extend(group[2:])
        else:
            result.append(37)  # %
            result.extend(group)

    return bytes(result)


def _url_encode_impl(
    obj: t.Union[t.Mapping[str, str], t.Iterable[t.Tuple[str, str]]],
    charset: str,
    sort: bool,
    key: t.Optional[t.Callable[[t.Tuple[str, str]], t.Any]],
) -> t.Iterator[str]:
    from .datastructures import iter_multi_items

    iterable: t.Iterable[t.Tuple[str, str]] = iter_multi_items(obj)

    if sort:
        iterable = sorted(iterable, key=key)

    for key_str, value_str in iterable:
        if value_str is None:
            continue

        if not isinstance(key_str, bytes):
            key_bytes = str(key_str).encode(charset)
        else:
            key_bytes = key_str

        if not isinstance(value_str, bytes):
            value_bytes = str(value_str).encode(charset)
        else:
            value_bytes = value_str

        yield f"{_fast_url_quote_plus(key_bytes)}={_fast_url_quote_plus(value_bytes)}"


def _url_unquote_legacy(value: str, unsafe: str = "") -> str:
    try:
        return url_unquote(value, charset="utf-8", errors="strict", unsafe=unsafe)
    except UnicodeError:
        return url_unquote(value, charset="latin1", unsafe=unsafe)


def url_parse(
    url: str, scheme: t.Optional[str] = None, allow_fragments: bool = True
) -> BaseURL:
    """Parses a URL from a string into a :class:`URL` tuple.  If the URL
    is lacking a scheme it can be provided as second argument. Otherwise,
    it is ignored.  Optionally fragments can be stripped from the URL
    by setting `allow_fragments` to `False`.

    The inverse of this function is :func:`url_unparse`.

    :param url: the URL to parse.
    :param scheme: the default schema to use if the URL is schemaless.
    :param allow_fragments: if set to `False` a fragment will be removed
                            from the URL.
    """
    s = _make_encode_wrapper(url)
    is_text_based = isinstance(url, str)

    if scheme is None:
        scheme = s("")
    netloc = query = fragment = s("")
    i = url.find(s(":"))
    if i > 0 and _scheme_re.match(_to_str(url[:i], errors="replace")):
        # make sure "iri" is not actually a port number (in which case
        # "scheme" is really part of the path)
        rest = url[i + 1 :]
        if not rest or any(c not in s("0123456789") for c in rest):
            # not a port number
            scheme, url = url[:i].lower(), rest

    if url[:2] == s("//"):
        delim = len(url)
        for c in s("/?#"):
            wdelim = url.find(c, 2)
            if wdelim >= 0:
                delim = min(delim, wdelim)
        netloc, url = url[2:delim], url[delim:]
        if (s("[") in netloc and s("]") not in netloc) or (
            s("]") in netloc and s("[") not in netloc
        ):
            raise ValueError("Invalid IPv6 URL")

    if allow_fragments and s("#") in url:
        url, fragment = url.split(s("#"), 1)
    if s("?") in url:
        url, query = url.split(s("?"), 1)

    result_type = URL if is_text_based else BytesURL
    return result_type(scheme, netloc, url, query, fragment)


def _make_fast_url_quote(
    charset: str = "utf-8",
    errors: str = "strict",
    safe: t.Union[str, bytes] = "/:",
    unsafe: t.Union[str, bytes] = "",
) -> t.Callable[[bytes], str]:
    """Precompile the translation table for a URL encoding function.

    Unlike :func:`url_quote`, the generated function only takes the
    string to quote.

    :param charset: The charset to encode the result with.
    :param errors: How to handle encoding errors.
    :param safe: An optional sequence of safe characters to never encode.
    :param unsafe: An optional sequence of unsafe characters to always encode.
    """
    if isinstance(safe, str):
        safe = safe.encode(charset, errors)

    if isinstance(unsafe, str):
        unsafe = unsafe.encode(charset, errors)

    safe = (frozenset(bytearray(safe)) | _always_safe) - frozenset(bytearray(unsafe))
    table = [chr(c) if c in safe else f"%{c:02X}" for c in range(256)]

    def quote(string: bytes) -> str:
        return "".join([table[c] for c in string])

    return quote


_fast_url_quote = _make_fast_url_quote()
_fast_quote_plus = _make_fast_url_quote(safe=" ", unsafe="+")


def _fast_url_quote_plus(string: bytes) -> str:
    return _fast_quote_plus(string).replace(" ", "+")


def url_quote(
    string: t.Union[str, bytes],
    charset: str = "utf-8",
    errors: str = "strict",
    safe: t.Union[str, bytes] = "/:",
    unsafe: t.Union[str, bytes] = "",
) -> str:
    """URL encode a single string with a given encoding.

    :param s: the string to quote.
    :param charset: the charset to be used.
    :param safe: an optional sequence of safe characters.
    :param unsafe: an optional sequence of unsafe characters.

    .. versionadded:: 0.9.2
       The `unsafe` parameter was added.
    """
    if not isinstance(string, (str, bytes, bytearray)):
        string = str(string)
    if isinstance(string, str):
        string = string.encode(charset, errors)
    if isinstance(safe, str):
        safe = safe.encode(charset, errors)
    if isinstance(unsafe, str):
        unsafe = unsafe.encode(charset, errors)
    safe = (frozenset(bytearray(safe)) | _always_safe) - frozenset(bytearray(unsafe))
    rv = bytearray()
    for char in bytearray(string):
        if char in safe:
            rv.append(char)
        else:
            rv.extend(_bytetohex[char])
    return bytes(rv).decode(charset)


def url_quote_plus(
    string: str, charset: str = "utf-8", errors: str = "strict", safe: str = ""
) -> str:
    """URL encode a single string with the given encoding and convert
    whitespace to "+".

    :param s: The string to quote.
    :param charset: The charset to be used.
    :param safe: An optional sequence of safe characters.
    """
    return url_quote(string, charset, errors, safe + " ", "+").replace(" ", "+")


def url_unparse(components: t.Tuple[str, str, str, str, str]) -> str:
    """The reverse operation to :meth:`url_parse`.  This accepts arbitrary
    as well as :class:`URL` tuples and returns a URL as a string.

    :param components: the parsed URL as tuple which should be converted
                       into a URL string.
    """
    _check_str_tuple(components)
    scheme, netloc, path, query, fragment = components
    s = _make_encode_wrapper(scheme)
    url = s("")

    # We generally treat file:///x and file:/x the same which is also
    # what browsers seem to do.  This also allows us to ignore a schema
    # register for netloc utilization or having to differentiate between
    # empty and missing netloc.
    if netloc or (scheme and path.startswith(s("/"))):
        if path and path[:1] != s("/"):
            path = s("/") + path
        url = s("//") + (netloc or s("")) + path
    elif path:
        url += path
    if scheme:
        url = scheme + s(":") + url
    if query:
        url = url + s("?") + query
    if fragment:
        url = url + s("#") + fragment
    return url


def url_unquote(
    s: t.Union[str, bytes],
    charset: str = "utf-8",
    errors: str = "replace",
    unsafe: str = "",
) -> str:
    """URL decode a single string with a given encoding.  If the charset
    is set to `None` no decoding is performed and raw bytes are
    returned.

    :param s: the string to unquote.
    :param charset: the charset of the query string.  If set to `None`
        no decoding will take place.
    :param errors: the error handling for the charset decoding.
    """
    rv = _unquote_to_bytes(s, unsafe)
    if charset is None:
        return rv
    return rv.decode(charset, errors)


def url_unquote_plus(
    s: t.Union[str, bytes], charset: str = "utf-8", errors: str = "replace"
) -> str:
    """URL decode a single string with the given `charset` and decode "+" to
    whitespace.

    Per default encoding errors are ignored.  If you want a different behavior
    you can set `errors` to ``'replace'`` or ``'strict'``.

    :param s: The string to unquote.
    :param charset: the charset of the query string.  If set to `None`
        no decoding will take place.
    :param errors: The error handling for the `charset` decoding.
    """
    if isinstance(s, str):
        s = s.replace("+", " ")
    else:
        s = s.replace(b"+", b" ")
    return url_unquote(s, charset, errors)


def url_fix(s: str, charset: str = "utf-8") -> str:
    r"""Sometimes you get an URL by a user that just isn't a real URL because
    it contains unsafe characters like ' ' and so on. This function can fix
    some of the problems in a similar way browsers handle data entered by the
    user:

    >>> url_fix('http://de.wikipedia.org/wiki/Elf (Begriffskl\xe4rung)')
    'http://de.wikipedia.org/wiki/Elf%20(Begriffskl%C3%A4rung)'

    :param s: the string with the URL to fix.
    :param charset: The target charset for the URL if the url was given
        as a string.
    """
    # First step is to switch to text processing and to convert
    # backslashes (which are invalid in URLs anyways) to slashes.  This is
    # consistent with what Chrome does.
    s = _to_str(s, charset, "replace").replace("\\", "/")

    # For the specific case that we look like a malformed windows URL
    # we want to fix this up manually:
    if s.startswith("file://") and s[7:8].isalpha() and s[8:10] in (":/", "|/"):
        s = f"file:///{s[7:]}"

    url = url_parse(s)
    path = url_quote(url.path, charset, safe="/%+$!*'(),")
    qs = url_quote_plus(url.query, charset, safe=":&%=+$!*'(),")
    anchor = url_quote_plus(url.fragment, charset, safe=":&%=+$!*'(),")
    return url_unparse((url.scheme, url.encode_netloc(), path, qs, anchor))


# not-unreserved characters remain quoted when unquoting to IRI
_to_iri_unsafe = "".join([chr(c) for c in range(128) if c not in _always_safe])


def _codec_error_url_quote(e: UnicodeError) -> t.Tuple[str, int]:
    """Used in :func:`uri_to_iri` after unquoting to re-quote any
    invalid bytes.
    """
    # the docs state that UnicodeError does have these attributes,
    # but mypy isn't picking them up
    out = _fast_url_quote(e.object[e.start : e.end])  # type: ignore
    return out, e.end  # type: ignore


codecs.register_error("werkzeug.url_quote", _codec_error_url_quote)


def uri_to_iri(
    uri: t.Union[str, t.Tuple[str, str, str, str, str]],
    charset: str = "utf-8",
    errors: str = "werkzeug.url_quote",
) -> str:
    """Convert a URI to an IRI. All valid UTF-8 characters are unquoted,
    leaving all reserved and invalid characters quoted. If the URL has
    a domain, it is decoded from Punycode.

    >>> uri_to_iri("http://xn--n3h.net/p%C3%A5th?q=%C3%A8ry%DF")
    'http://\\u2603.net/p\\xe5th?q=\\xe8ry%DF'

    :param uri: The URI to convert.
    :param charset: The encoding to encode unquoted bytes with.
    :param errors: Error handler to use during ``bytes.encode``. By
        default, invalid bytes are left quoted.

    .. versionchanged:: 0.15
        All reserved and invalid characters remain quoted. Previously,
        only some reserved characters were preserved, and invalid bytes
        were replaced instead of left quoted.

    .. versionadded:: 0.6
    """
    if isinstance(uri, tuple):
        uri = url_unparse(uri)

    uri = url_parse(_to_str(uri, charset))
    path = url_unquote(uri.path, charset, errors, _to_iri_unsafe)
    query = url_unquote(uri.query, charset, errors, _to_iri_unsafe)
    fragment = url_unquote(uri.fragment, charset, errors, _to_iri_unsafe)
    return url_unparse((uri.scheme, uri.decode_netloc(), path, query, fragment))


# reserved characters remain unquoted when quoting to URI
_to_uri_safe = ":/?#[]@!$&'()*+,;=%"


def iri_to_uri(
    iri: t.Union[str, t.Tuple[str, str, str, str, str]],
    charset: str = "utf-8",
    errors: str = "strict",
    safe_conversion: bool = False,
) -> str:
    """Convert an IRI to a URI. All non-ASCII and unsafe characters are
    quoted. If the URL has a domain, it is encoded to Punycode.

    >>> iri_to_uri('http://\\u2603.net/p\\xe5th?q=\\xe8ry%DF')
    'http://xn--n3h.net/p%C3%A5th?q=%C3%A8ry%DF'

    :param iri: The IRI to convert.
    :param charset: The encoding of the IRI.
    :param errors: Error handler to use during ``bytes.encode``.
    :param safe_conversion: Return the URL unchanged if it only contains
        ASCII characters and no whitespace. See the explanation below.

    There is a general problem with IRI conversion with some protocols
    that are in violation of the URI specification. Consider the
    following two IRIs::

        magnet:?xt=uri:whatever
        itms-services://?action=download-manifest

    After parsing, we don't know if the scheme requires the ``//``,
    which is dropped if empty, but conveys different meanings in the
    final URL if it's present or not. In this case, you can use
    ``safe_conversion``, which will return the URL unchanged if it only
    contains ASCII characters and no whitespace. This can result in a
    URI with unquoted characters if it was not already quoted correctly,
    but preserves the URL's semantics. Werkzeug uses this for the
    ``Location`` header for redirects.

    .. versionchanged:: 0.15
        All reserved characters remain unquoted. Previously, only some
        reserved characters were left unquoted.

    .. versionchanged:: 0.9.6
       The ``safe_conversion`` parameter was added.

    .. versionadded:: 0.6
    """
    if isinstance(iri, tuple):
        iri = url_unparse(iri)

    if safe_conversion:
        # If we're not sure if it's safe to convert the URL, and it only
        # contains ASCII characters, return it unconverted.
        try:
            native_iri = _to_str(iri)
            ascii_iri = native_iri.encode("ascii")

            # Only return if it doesn't have whitespace. (Why?)
            if len(ascii_iri.split()) == 1:
                return native_iri
        except UnicodeError:
            pass

    iri = url_parse(_to_str(iri, charset, errors))
    path = url_quote(iri.path, charset, errors, _to_uri_safe)
    query = url_quote(iri.query, charset, errors, _to_uri_safe)
    fragment = url_quote(iri.fragment, charset, errors, _to_uri_safe)
    return url_unparse((iri.scheme, iri.encode_netloc(), path, query, fragment))


def url_decode(
    s: t.AnyStr,
    charset: str = "utf-8",
    decode_keys: None = None,
    include_empty: bool = True,
    errors: str = "replace",
    separator: str = "&",
    cls: t.Optional[t.Type["ds.MultiDict"]] = None,
) -> "ds.MultiDict[str, str]":
    """Parse a query string and return it as a :class:`MultiDict`.

    :param s: The query string to parse.
    :param charset: Decode bytes to string with this charset. If not
        given, bytes are returned as-is.
    :param include_empty: Include keys with empty values in the dict.
    :param errors: Error handling behavior when decoding bytes.
    :param separator: Separator character between pairs.
    :param cls: Container to hold result instead of :class:`MultiDict`.

    .. versionchanged:: 2.0
        The ``decode_keys`` parameter is deprecated and will be removed
        in Werkzeug 2.1.

    .. versionchanged:: 0.5
        In previous versions ";" and "&" could be used for url decoding.
        Now only "&" is supported. If you want to use ";", a different
        ``separator`` can be provided.

    .. versionchanged:: 0.5
        The ``cls`` parameter was added.
    """
    if decode_keys is not None:
        warnings.warn(
            "'decode_keys' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
    if cls is None:
        from .datastructures import MultiDict  # noqa: F811

        cls = MultiDict
    if isinstance(s, str) and not isinstance(separator, str):
        separator = separator.decode(charset or "ascii")
    elif isinstance(s, bytes) and not isinstance(separator, bytes):
        separator = separator.encode(charset or "ascii")  # type: ignore
    return cls(
        _url_decode_impl(
            s.split(separator), charset, include_empty, errors  # type: ignore
        )
    )


def url_decode_stream(
    stream: t.IO[bytes],
    charset: str = "utf-8",
    decode_keys: None = None,
    include_empty: bool = True,
    errors: str = "replace",
    separator: bytes = b"&",
    cls: t.Optional[t.Type["ds.MultiDict"]] = None,
    limit: t.Optional[int] = None,
    return_iterator: bool = False,
) -> "ds.MultiDict[str, str]":
    """Works like :func:`url_decode` but decodes a stream.  The behavior
    of stream and limit follows functions like
    :func:`~werkzeug.wsgi.make_line_iter`.  The generator of pairs is
    directly fed to the `cls` so you can consume the data while it's
    parsed.

    :param stream: a stream with the encoded querystring
    :param charset: the charset of the query string.  If set to `None`
        no decoding will take place.
    :param include_empty: Set to `False` if you don't want empty values to
                          appear in the dict.
    :param errors: the decoding error behavior.
    :param separator: the pair separator to be used, defaults to ``&``
    :param cls: an optional dict class to use.  If this is not specified
                       or `None` the default :class:`MultiDict` is used.
    :param limit: the content length of the URL data.  Not necessary if
                  a limited stream is provided.

    .. versionchanged:: 2.0
        The ``decode_keys`` and ``return_iterator`` parameters are
        deprecated and will be removed in Werkzeug 2.1.

    .. versionadded:: 0.8
    """
    from .wsgi import make_chunk_iter

    if decode_keys is not None:
        warnings.warn(
            "'decode_keys' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )

    pair_iter = make_chunk_iter(stream, separator, limit)
    decoder = _url_decode_impl(pair_iter, charset, include_empty, errors)

    if return_iterator:
        warnings.warn(
            "'return_iterator' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
        return decoder  # type: ignore

    if cls is None:
        from .datastructures import MultiDict  # noqa: F811

        cls = MultiDict

    return cls(decoder)


def _url_decode_impl(
    pair_iter: t.Iterable[t.AnyStr], charset: str, include_empty: bool, errors: str
) -> t.Iterator[t.Tuple[str, str]]:
    for pair in pair_iter:
        if not pair:
            continue
        s = _make_encode_wrapper(pair)
        equal = s("=")
        if equal in pair:
            key, value = pair.split(equal, 1)
        else:
            if not include_empty:
                continue
            key = pair
            value = s("")
        yield (
            url_unquote_plus(key, charset, errors),
            url_unquote_plus(value, charset, errors),
        )


def url_encode(
    obj: t.Union[t.Mapping[str, str], t.Iterable[t.Tuple[str, str]]],
    charset: str = "utf-8",
    encode_keys: None = None,
    sort: bool = False,
    key: t.Optional[t.Callable[[t.Tuple[str, str]], t.Any]] = None,
    separator: str = "&",
) -> str:
    """URL encode a dict/`MultiDict`.  If a value is `None` it will not appear
    in the result string.  Per default only values are encoded into the target
    charset strings.

    :param obj: the object to encode into a query string.
    :param charset: the charset of the query string.
    :param sort: set to `True` if you want parameters to be sorted by `key`.
    :param separator: the separator to be used for the pairs.
    :param key: an optional function to be used for sorting.  For more details
                check out the :func:`sorted` documentation.

    .. versionchanged:: 2.0
        The ``encode_keys`` parameter is deprecated and will be removed
        in Werkzeug 2.1.

    .. versionchanged:: 0.5
        Added the ``sort``, ``key``, and ``separator`` parameters.
    """
    if encode_keys is not None:
        warnings.warn(
            "'encode_keys' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
    separator = _to_str(separator, "ascii")
    return separator.join(_url_encode_impl(obj, charset, sort, key))


def url_encode_stream(
    obj: t.Union[t.Mapping[str, str], t.Iterable[t.Tuple[str, str]]],
    stream: t.Optional[t.IO[str]] = None,
    charset: str = "utf-8",
    encode_keys: None = None,
    sort: bool = False,
    key: t.Optional[t.Callable[[t.Tuple[str, str]], t.Any]] = None,
    separator: str = "&",
) -> None:
    """Like :meth:`url_encode` but writes the results to a stream
    object.  If the stream is `None` a generator over all encoded
    pairs is returned.

    :param obj: the object to encode into a query string.
    :param stream: a stream to write the encoded object into or `None` if
                   an iterator over the encoded pairs should be returned.  In
                   that case the separator argument is ignored.
    :param charset: the charset of the query string.
    :param sort: set to `True` if you want parameters to be sorted by `key`.
    :param separator: the separator to be used for the pairs.
    :param key: an optional function to be used for sorting.  For more details
                check out the :func:`sorted` documentation.

    .. versionchanged:: 2.0
        The ``encode_keys`` parameter is deprecated and will be removed
        in Werkzeug 2.1.

    .. versionadded:: 0.8
    """
    if encode_keys is not None:
        warnings.warn(
            "'encode_keys' is deprecated and will be removed in Werkzeug 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
    separator = _to_str(separator, "ascii")
    gen = _url_encode_impl(obj, charset, sort, key)
    if stream is None:
        return gen  # type: ignore
    for idx, chunk in enumerate(gen):
        if idx:
            stream.write(separator)
        stream.write(chunk)
    return None


def url_join(
    base: t.Union[str, t.Tuple[str, str, str, str, str]],
    url: t.Union[str, t.Tuple[str, str, str, str, str]],
    allow_fragments: bool = True,
) -> str:
    """Join a base URL and a possibly relative URL to form an absolute
    interpretation of the latter.

    :param base: the base URL for the join operation.
    :param url: the URL to join.
    :param allow_fragments: indicates whether fragments should be allowed.
    """
    if isinstance(base, tuple):
        base = url_unparse(base)
    if isinstance(url, tuple):
        url = url_unparse(url)

    _check_str_tuple((base, url))
    s = _make_encode_wrapper(base)

    if not base:
        return url
    if not url:
        return base

    bscheme, bnetloc, bpath, bquery, bfragment = url_parse(
        base, allow_fragments=allow_fragments
    )
    scheme, netloc, path, query, fragment = url_parse(url, bscheme, allow_fragments)
    if scheme != bscheme:
        return url
    if netloc:
        return url_unparse((scheme, netloc, path, query, fragment))
    netloc = bnetloc

    if path[:1] == s("/"):
        segments = path.split(s("/"))
    elif not path:
        segments = bpath.split(s("/"))
        if not query:
            query = bquery
    else:
        segments = bpath.split(s("/"))[:-1] + path.split(s("/"))

    # If the rightmost part is "./" we want to keep the slash but
    # remove the dot.
    if segments[-1] == s("."):
        segments[-1] = s("")

    # Resolve ".." and "."
    segments = [segment for segment in segments if segment != s(".")]
    while True:
        i = 1
        n = len(segments) - 1
        while i < n:
            if segments[i] == s("..") and segments[i - 1] not in (s(""), s("..")):
                del segments[i - 1 : i + 1]
                break
            i += 1
        else:
            break

    # Remove trailing ".." if the URL is absolute
    unwanted_marker = [s(""), s("..")]
    while segments[:2] == unwanted_marker:
        del segments[1]

    path = s("/").join(segments)
    return url_unparse((scheme, netloc, path, query, fragment))


class Href:
    """Implements a callable that constructs URLs with the given base. The
    function can be called with any number of positional and keyword
    arguments which than are used to assemble the URL.  Works with URLs
    and posix paths.

    Positional arguments are appended as individual segments to
    the path of the URL:

    >>> href = Href('/foo')
    >>> href('bar', 23)
    '/foo/bar/23'
    >>> href('foo', bar=23)
    '/foo/foo?bar=23'

    If any of the arguments (positional or keyword) evaluates to `None` it
    will be skipped.  If no keyword arguments are given the last argument
    can be a :class:`dict` or :class:`MultiDict` (or any other dict subclass),
    otherwise the keyword arguments are used for the query parameters, cutting
    off the first trailing underscore of the parameter name:

    >>> href(is_=42)
    '/foo?is=42'
    >>> href({'foo': 'bar'})
    '/foo?foo=bar'

    Combining of both methods is not allowed:

    >>> href({'foo': 'bar'}, bar=42)
    Traceback (most recent call last):
      ...
    TypeError: keyword arguments and query-dicts can't be combined

    Accessing attributes on the href object creates a new href object with
    the attribute name as prefix:

    >>> bar_href = href.bar
    >>> bar_href("blub")
    '/foo/bar/blub'

    If `sort` is set to `True` the items are sorted by `key` or the default
    sorting algorithm:

    >>> href = Href("/", sort=True)
    >>> href(a=1, b=2, c=3)
    '/?a=1&b=2&c=3'

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use :mod:`werkzeug.routing`
        instead.

    .. versionadded:: 0.5
        `sort` and `key` were added.
    """

    def __init__(  # type: ignore
        self, base="./", charset="utf-8", sort=False, key=None
    ):
        warnings.warn(
            "'Href' is deprecated and will be removed in Werkzeug 2.1."
            " Use 'werkzeug.routing' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if not base:
            base = "./"
        self.base = base
        self.charset = charset
        self.sort = sort
        self.key = key

    def __getattr__(self, name):  # type: ignore
        if name[:2] == "__":
            raise AttributeError(name)
        base = self.base
        if base[-1:] != "/":
            base += "/"
        return Href(url_join(base, name), self.charset, self.sort, self.key)

    def __call__(self, *path, **query):  # type: ignore
        if path and isinstance(path[-1], dict):
            if query:
                raise TypeError("keyword arguments and query-dicts can't be combined")
            query, path = path[-1], path[:-1]
        elif query:
            query = {k[:-1] if k.endswith("_") else k: v for k, v in query.items()}
        path = "/".join(
            [
                _to_str(url_quote(x, self.charset), "ascii")
                for x in path
                if x is not None
            ]
        ).lstrip("/")
        rv = self.base
        if path:
            if not rv.endswith("/"):
                rv += "/"
            rv = url_join(rv, f"./{path}")
        if query:
            rv += "?" + _to_str(
                url_encode(query, self.charset, sort=self.sort, key=self.key), "ascii"
            )
        return rv
