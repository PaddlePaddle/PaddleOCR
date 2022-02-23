import base64
import email.utils
import re
import typing
import typing as t
import warnings
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from datetime import timezone
from enum import Enum
from hashlib import sha1
from time import mktime
from time import struct_time
from urllib.parse import unquote_to_bytes as _unquote
from urllib.request import parse_http_list as _parse_list_header

from ._internal import _cookie_parse_impl
from ._internal import _cookie_quote
from ._internal import _make_cookie_domain
from ._internal import _to_bytes
from ._internal import _to_str
from ._internal import _wsgi_decoding_dance
from werkzeug._internal import _dt_as_utc

if t.TYPE_CHECKING:
    import typing_extensions as te
    from _typeshed.wsgi import WSGIEnvironment

# for explanation of "media-range", etc. see Sections 5.3.{1,2} of RFC 7231
_accept_re = re.compile(
    r"""
    (                       # media-range capturing-parenthesis
      [^\s;,]+              # type/subtype
      (?:[ \t]*;[ \t]*      # ";"
        (?:                 # parameter non-capturing-parenthesis
          [^\s;,q][^\s;,]*  # token that doesn't start with "q"
        |                   # or
          q[^\s;,=][^\s;,]* # token that is more than just "q"
        )
      )*                    # zero or more parameters
    )                       # end of media-range
    (?:[ \t]*;[ \t]*q=      # weight is a "q" parameter
      (\d*(?:\.\d+)?)       # qvalue capturing-parentheses
      [^,]*                 # "extension" accept params: who cares?
    )?                      # accept params are optional
    """,
    re.VERBOSE,
)
_token_chars = frozenset(
    "!#$%&'*+-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ^_`abcdefghijklmnopqrstuvwxyz|~"
)
_etag_re = re.compile(r'([Ww]/)?(?:"(.*?)"|(.*?))(?:\s*,\s*|$)')
_option_header_piece_re = re.compile(
    r"""
    ;\s*,?\s*  # newlines were replaced with commas
    (?P<key>
        "[^"\\]*(?:\\.[^"\\]*)*"  # quoted string
    |
        [^\s;,=*]+  # token
    )
    (?:\*(?P<count>\d+))?  # *1, optional continuation index
    \s*
    (?:  # optionally followed by =value
        (?:  # equals sign, possibly with encoding
            \*\s*=\s*  # * indicates extended notation
            (?:  # optional encoding
                (?P<encoding>[^\s]+?)
                '(?P<language>[^\s]*?)'
            )?
        |
            =\s*  # basic notation
        )
        (?P<value>
            "[^"\\]*(?:\\.[^"\\]*)*"  # quoted string
        |
            [^;,]+  # token
        )?
    )?
    \s*
    """,
    flags=re.VERBOSE,
)
_option_header_start_mime_type = re.compile(r",\s*([^;,\s]+)([;,]\s*.+)?")
_entity_headers = frozenset(
    [
        "allow",
        "content-encoding",
        "content-language",
        "content-length",
        "content-location",
        "content-md5",
        "content-range",
        "content-type",
        "expires",
        "last-modified",
    ]
)
_hop_by_hop_headers = frozenset(
    [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
)
HTTP_STATUS_CODES = {
    100: "Continue",
    101: "Switching Protocols",
    102: "Processing",
    103: "Early Hints",  # see RFC 8297
    200: "OK",
    201: "Created",
    202: "Accepted",
    203: "Non Authoritative Information",
    204: "No Content",
    205: "Reset Content",
    206: "Partial Content",
    207: "Multi Status",
    208: "Already Reported",  # see RFC 5842
    226: "IM Used",  # see RFC 3229
    300: "Multiple Choices",
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified",
    305: "Use Proxy",
    306: "Switch Proxy",  # unused
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request",
    401: "Unauthorized",
    402: "Payment Required",  # unused
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    407: "Proxy Authentication Required",
    408: "Request Timeout",
    409: "Conflict",
    410: "Gone",
    411: "Length Required",
    412: "Precondition Failed",
    413: "Request Entity Too Large",
    414: "Request URI Too Long",
    415: "Unsupported Media Type",
    416: "Requested Range Not Satisfiable",
    417: "Expectation Failed",
    418: "I'm a teapot",  # see RFC 2324
    421: "Misdirected Request",  # see RFC 7540
    422: "Unprocessable Entity",
    423: "Locked",
    424: "Failed Dependency",
    425: "Too Early",  # see RFC 8470
    426: "Upgrade Required",
    428: "Precondition Required",  # see RFC 6585
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    449: "Retry With",  # proprietary MS extension
    451: "Unavailable For Legal Reasons",
    500: "Internal Server Error",
    501: "Not Implemented",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
    505: "HTTP Version Not Supported",
    506: "Variant Also Negotiates",  # see RFC 2295
    507: "Insufficient Storage",
    508: "Loop Detected",  # see RFC 5842
    510: "Not Extended",
    511: "Network Authentication Failed",
}


class COEP(Enum):
    """Cross Origin Embedder Policies"""

    UNSAFE_NONE = "unsafe-none"
    REQUIRE_CORP = "require-corp"


class COOP(Enum):
    """Cross Origin Opener Policies"""

    UNSAFE_NONE = "unsafe-none"
    SAME_ORIGIN_ALLOW_POPUPS = "same-origin-allow-popups"
    SAME_ORIGIN = "same-origin"


def quote_header_value(
    value: t.Union[str, int], extra_chars: str = "", allow_token: bool = True
) -> str:
    """Quote a header value if necessary.

    .. versionadded:: 0.5

    :param value: the value to quote.
    :param extra_chars: a list of extra characters to skip quoting.
    :param allow_token: if this is enabled token values are returned
                        unchanged.
    """
    if isinstance(value, bytes):
        value = value.decode("latin1")
    value = str(value)
    if allow_token:
        token_chars = _token_chars | set(extra_chars)
        if set(value).issubset(token_chars):
            return value
    value = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{value}"'


def unquote_header_value(value: str, is_filename: bool = False) -> str:
    r"""Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.

    .. versionadded:: 0.5

    :param value: the header value to unquote.
    :param is_filename: The value represents a filename or path.
    """
    if value and value[0] == value[-1] == '"':
        # this is not the real unquoting, but fixing this so that the
        # RFC is met will result in bugs with internet explorer and
        # probably some other browsers as well.  IE for example is
        # uploading files with "C:\foo\bar.txt" as filename
        value = value[1:-1]

        # if this is a filename and the starting characters look like
        # a UNC path, then just return the value without quotes.  Using the
        # replace sequence below on a UNC path has the effect of turning
        # the leading double slash into a single slash and then
        # _fix_ie_filename() doesn't work correctly.  See #458.
        if not is_filename or value[:2] != "\\\\":
            return value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def dump_options_header(
    header: t.Optional[str], options: t.Mapping[str, t.Optional[t.Union[str, int]]]
) -> str:
    """The reverse function to :func:`parse_options_header`.

    :param header: the header to dump
    :param options: a dict of options to append.
    """
    segments = []
    if header is not None:
        segments.append(header)
    for key, value in options.items():
        if value is None:
            segments.append(key)
        else:
            segments.append(f"{key}={quote_header_value(value)}")
    return "; ".join(segments)


def dump_header(
    iterable: t.Union[t.Dict[str, t.Union[str, int]], t.Iterable[str]],
    allow_token: bool = True,
) -> str:
    """Dump an HTTP header again.  This is the reversal of
    :func:`parse_list_header`, :func:`parse_set_header` and
    :func:`parse_dict_header`.  This also quotes strings that include an
    equals sign unless you pass it as dict of key, value pairs.

    >>> dump_header({'foo': 'bar baz'})
    'foo="bar baz"'
    >>> dump_header(('foo', 'bar baz'))
    'foo, "bar baz"'

    :param iterable: the iterable or dict of values to quote.
    :param allow_token: if set to `False` tokens as values are disallowed.
                        See :func:`quote_header_value` for more details.
    """
    if isinstance(iterable, dict):
        items = []
        for key, value in iterable.items():
            if value is None:
                items.append(key)
            else:
                items.append(
                    f"{key}={quote_header_value(value, allow_token=allow_token)}"
                )
    else:
        items = [quote_header_value(x, allow_token=allow_token) for x in iterable]
    return ", ".join(items)


def dump_csp_header(header: "ds.ContentSecurityPolicy") -> str:
    """Dump a Content Security Policy header.

    These are structured into policies such as "default-src 'self';
    script-src 'self'".

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    """
    return "; ".join(f"{key} {value}" for key, value in header.items())


def parse_list_header(value: str) -> t.List[str]:
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    """
    result = []
    for item in _parse_list_header(value):
        if item[:1] == item[-1:] == '"':
            item = unquote_header_value(item[1:-1])
        result.append(item)
    return result


def parse_dict_header(value: str, cls: t.Type[dict] = dict) -> t.Dict[str, str]:
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict (or any other mapping object created from
    the type with a dict like interface provided by the `cls` argument):

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    .. versionchanged:: 0.9
       Added support for `cls` argument.

    :param value: a string with a dict header.
    :param cls: callable to use for storage of parsed results.
    :return: an instance of `cls`
    """
    result = cls()
    if isinstance(value, bytes):
        value = value.decode("latin1")
    for item in _parse_list_header(value):
        if "=" not in item:
            result[item] = None
            continue
        name, value = item.split("=", 1)
        if value[:1] == value[-1:] == '"':
            value = unquote_header_value(value[1:-1])
        result[name] = value
    return result


@typing.overload
def parse_options_header(
    value: t.Optional[str], multiple: "te.Literal[False]" = False
) -> t.Tuple[str, t.Dict[str, str]]:
    ...


@typing.overload
def parse_options_header(
    value: t.Optional[str], multiple: "te.Literal[True]"
) -> t.Tuple[t.Any, ...]:
    ...


def parse_options_header(
    value: t.Optional[str], multiple: bool = False
) -> t.Union[t.Tuple[str, t.Dict[str, str]], t.Tuple[t.Any, ...]]:
    """Parse a ``Content-Type`` like header into a tuple with the content
    type and the options:

    >>> parse_options_header('text/html; charset=utf8')
    ('text/html', {'charset': 'utf8'})

    This should not be used to parse ``Cache-Control`` like headers that use
    a slightly different format.  For these headers use the
    :func:`parse_dict_header` function.

    .. versionchanged:: 0.15
        :rfc:`2231` parameter continuations are handled.

    .. versionadded:: 0.5

    :param value: the header to parse.
    :param multiple: Whether try to parse and return multiple MIME types
    :return: (mimetype, options) or (mimetype, options, mimetype, options, …)
             if multiple=True
    """
    if not value:
        return "", {}

    result: t.List[t.Any] = []

    value = "," + value.replace("\n", ",")
    while value:
        match = _option_header_start_mime_type.match(value)
        if not match:
            break
        result.append(match.group(1))  # mimetype
        options: t.Dict[str, str] = {}
        # Parse options
        rest = match.group(2)
        encoding: t.Optional[str]
        continued_encoding: t.Optional[str] = None
        while rest:
            optmatch = _option_header_piece_re.match(rest)
            if not optmatch:
                break
            option, count, encoding, language, option_value = optmatch.groups()
            # Continuations don't have to supply the encoding after the
            # first line. If we're in a continuation, track the current
            # encoding to use for subsequent lines. Reset it when the
            # continuation ends.
            if not count:
                continued_encoding = None
            else:
                if not encoding:
                    encoding = continued_encoding
                continued_encoding = encoding
            option = unquote_header_value(option)
            if option_value is not None:
                option_value = unquote_header_value(option_value, option == "filename")
                if encoding is not None:
                    option_value = _unquote(option_value).decode(encoding)
            if count:
                # Continuations append to the existing value. For
                # simplicity, this ignores the possibility of
                # out-of-order indices, which shouldn't happen anyway.
                options[option] = options.get(option, "") + option_value
            else:
                options[option] = option_value
            rest = rest[optmatch.end() :]
        result.append(options)
        if multiple is False:
            return tuple(result)
        value = rest

    return tuple(result) if result else ("", {})


_TAnyAccept = t.TypeVar("_TAnyAccept", bound="ds.Accept")


@typing.overload
def parse_accept_header(value: t.Optional[str]) -> "ds.Accept":
    ...


@typing.overload
def parse_accept_header(
    value: t.Optional[str], cls: t.Type[_TAnyAccept]
) -> _TAnyAccept:
    ...


def parse_accept_header(
    value: t.Optional[str], cls: t.Optional[t.Type[_TAnyAccept]] = None
) -> _TAnyAccept:
    """Parses an HTTP Accept-* header.  This does not implement a complete
    valid algorithm but one that supports at least value and quality
    extraction.

    Returns a new :class:`Accept` object (basically a list of ``(value, quality)``
    tuples sorted by the quality with some additional accessor methods).

    The second parameter can be a subclass of :class:`Accept` that is created
    with the parsed values and returned.

    :param value: the accept header string to be parsed.
    :param cls: the wrapper class for the return value (can be
                         :class:`Accept` or a subclass thereof)
    :return: an instance of `cls`.
    """
    if cls is None:
        cls = t.cast(t.Type[_TAnyAccept], ds.Accept)

    if not value:
        return cls(None)

    result = []
    for match in _accept_re.finditer(value):
        quality_match = match.group(2)
        if not quality_match:
            quality: float = 1
        else:
            quality = max(min(float(quality_match), 1), 0)
        result.append((match.group(1), quality))
    return cls(result)


_TAnyCC = t.TypeVar("_TAnyCC", bound="ds._CacheControl")
_t_cc_update = t.Optional[t.Callable[[_TAnyCC], None]]


@typing.overload
def parse_cache_control_header(
    value: t.Optional[str], on_update: _t_cc_update, cls: None = None
) -> "ds.RequestCacheControl":
    ...


@typing.overload
def parse_cache_control_header(
    value: t.Optional[str], on_update: _t_cc_update, cls: t.Type[_TAnyCC]
) -> _TAnyCC:
    ...


def parse_cache_control_header(
    value: t.Optional[str],
    on_update: _t_cc_update = None,
    cls: t.Optional[t.Type[_TAnyCC]] = None,
) -> _TAnyCC:
    """Parse a cache control header.  The RFC differs between response and
    request cache control, this method does not.  It's your responsibility
    to not use the wrong control statements.

    .. versionadded:: 0.5
       The `cls` was added.  If not specified an immutable
       :class:`~werkzeug.datastructures.RequestCacheControl` is returned.

    :param value: a cache control header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.CacheControl`
                      object is changed.
    :param cls: the class for the returned object.  By default
                :class:`~werkzeug.datastructures.RequestCacheControl` is used.
    :return: a `cls` object.
    """
    if cls is None:
        cls = t.cast(t.Type[_TAnyCC], ds.RequestCacheControl)

    if not value:
        return cls((), on_update)

    return cls(parse_dict_header(value), on_update)


_TAnyCSP = t.TypeVar("_TAnyCSP", bound="ds.ContentSecurityPolicy")
_t_csp_update = t.Optional[t.Callable[[_TAnyCSP], None]]


@typing.overload
def parse_csp_header(
    value: t.Optional[str], on_update: _t_csp_update, cls: None = None
) -> "ds.ContentSecurityPolicy":
    ...


@typing.overload
def parse_csp_header(
    value: t.Optional[str], on_update: _t_csp_update, cls: t.Type[_TAnyCSP]
) -> _TAnyCSP:
    ...


def parse_csp_header(
    value: t.Optional[str],
    on_update: _t_csp_update = None,
    cls: t.Optional[t.Type[_TAnyCSP]] = None,
) -> _TAnyCSP:
    """Parse a Content Security Policy header.

    .. versionadded:: 1.0.0
       Support for Content Security Policy headers was added.

    :param value: a csp header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the object is changed.
    :param cls: the class for the returned object.  By default
                :class:`~werkzeug.datastructures.ContentSecurityPolicy` is used.
    :return: a `cls` object.
    """
    if cls is None:
        cls = t.cast(t.Type[_TAnyCSP], ds.ContentSecurityPolicy)

    if value is None:
        return cls((), on_update)

    items = []

    for policy in value.split(";"):
        policy = policy.strip()

        # Ignore badly formatted policies (no space)
        if " " in policy:
            directive, value = policy.strip().split(" ", 1)
            items.append((directive.strip(), value.strip()))

    return cls(items, on_update)


def parse_set_header(
    value: t.Optional[str],
    on_update: t.Optional[t.Callable[["ds.HeaderSet"], None]] = None,
) -> "ds.HeaderSet":
    """Parse a set-like header and return a
    :class:`~werkzeug.datastructures.HeaderSet` object:

    >>> hs = parse_set_header('token, "quoted value"')

    The return value is an object that treats the items case-insensitively
    and keeps the order of the items:

    >>> 'TOKEN' in hs
    True
    >>> hs.index('quoted value')
    1
    >>> hs
    HeaderSet(['token', 'quoted value'])

    To create a header from the :class:`HeaderSet` again, use the
    :func:`dump_header` function.

    :param value: a set header to be parsed.
    :param on_update: an optional callable that is called every time a
                      value on the :class:`~werkzeug.datastructures.HeaderSet`
                      object is changed.
    :return: a :class:`~werkzeug.datastructures.HeaderSet`
    """
    if not value:
        return ds.HeaderSet(None, on_update)
    return ds.HeaderSet(parse_list_header(value), on_update)


def parse_authorization_header(
    value: t.Optional[str],
) -> t.Optional["ds.Authorization"]:
    """Parse an HTTP basic/digest authorization header transmitted by the web
    browser.  The return value is either `None` if the header was invalid or
    not given, otherwise an :class:`~werkzeug.datastructures.Authorization`
    object.

    :param value: the authorization header to parse.
    :return: a :class:`~werkzeug.datastructures.Authorization` object or `None`.
    """
    if not value:
        return None
    value = _wsgi_decoding_dance(value)
    try:
        auth_type, auth_info = value.split(None, 1)
        auth_type = auth_type.lower()
    except ValueError:
        return None
    if auth_type == "basic":
        try:
            username, password = base64.b64decode(auth_info).split(b":", 1)
        except Exception:
            return None
        try:
            return ds.Authorization(
                "basic",
                {
                    "username": _to_str(username, "utf-8"),
                    "password": _to_str(password, "utf-8"),
                },
            )
        except UnicodeDecodeError:
            return None
    elif auth_type == "digest":
        auth_map = parse_dict_header(auth_info)
        for key in "username", "realm", "nonce", "uri", "response":
            if key not in auth_map:
                return None
        if "qop" in auth_map:
            if not auth_map.get("nc") or not auth_map.get("cnonce"):
                return None
        return ds.Authorization("digest", auth_map)
    return None


def parse_www_authenticate_header(
    value: t.Optional[str],
    on_update: t.Optional[t.Callable[["ds.WWWAuthenticate"], None]] = None,
) -> "ds.WWWAuthenticate":
    """Parse an HTTP WWW-Authenticate header into a
    :class:`~werkzeug.datastructures.WWWAuthenticate` object.

    :param value: a WWW-Authenticate header to parse.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.WWWAuthenticate`
                      object is changed.
    :return: a :class:`~werkzeug.datastructures.WWWAuthenticate` object.
    """
    if not value:
        return ds.WWWAuthenticate(on_update=on_update)
    try:
        auth_type, auth_info = value.split(None, 1)
        auth_type = auth_type.lower()
    except (ValueError, AttributeError):
        return ds.WWWAuthenticate(value.strip().lower(), on_update=on_update)
    return ds.WWWAuthenticate(auth_type, parse_dict_header(auth_info), on_update)


def parse_if_range_header(value: t.Optional[str]) -> "ds.IfRange":
    """Parses an if-range header which can be an etag or a date.  Returns
    a :class:`~werkzeug.datastructures.IfRange` object.

    .. versionchanged:: 2.0
        If the value represents a datetime, it is timezone-aware.

    .. versionadded:: 0.7
    """
    if not value:
        return ds.IfRange()
    date = parse_date(value)
    if date is not None:
        return ds.IfRange(date=date)
    # drop weakness information
    return ds.IfRange(unquote_etag(value)[0])


def parse_range_header(
    value: t.Optional[str], make_inclusive: bool = True
) -> t.Optional["ds.Range"]:
    """Parses a range header into a :class:`~werkzeug.datastructures.Range`
    object.  If the header is missing or malformed `None` is returned.
    `ranges` is a list of ``(start, stop)`` tuples where the ranges are
    non-inclusive.

    .. versionadded:: 0.7
    """
    if not value or "=" not in value:
        return None

    ranges = []
    last_end = 0
    units, rng = value.split("=", 1)
    units = units.strip().lower()

    for item in rng.split(","):
        item = item.strip()
        if "-" not in item:
            return None
        if item.startswith("-"):
            if last_end < 0:
                return None
            try:
                begin = int(item)
            except ValueError:
                return None
            end = None
            last_end = -1
        elif "-" in item:
            begin_str, end_str = item.split("-", 1)
            begin_str = begin_str.strip()
            end_str = end_str.strip()
            if not begin_str.isdigit():
                return None
            begin = int(begin_str)
            if begin < last_end or last_end < 0:
                return None
            if end_str:
                if not end_str.isdigit():
                    return None
                end = int(end_str) + 1
                if begin >= end:
                    return None
            else:
                end = None
            last_end = end if end is not None else -1
        ranges.append((begin, end))

    return ds.Range(units, ranges)


def parse_content_range_header(
    value: t.Optional[str],
    on_update: t.Optional[t.Callable[["ds.ContentRange"], None]] = None,
) -> t.Optional["ds.ContentRange"]:
    """Parses a range header into a
    :class:`~werkzeug.datastructures.ContentRange` object or `None` if
    parsing is not possible.

    .. versionadded:: 0.7

    :param value: a content range header to be parsed.
    :param on_update: an optional callable that is called every time a value
                      on the :class:`~werkzeug.datastructures.ContentRange`
                      object is changed.
    """
    if value is None:
        return None
    try:
        units, rangedef = (value or "").strip().split(None, 1)
    except ValueError:
        return None

    if "/" not in rangedef:
        return None
    rng, length_str = rangedef.split("/", 1)
    if length_str == "*":
        length = None
    elif length_str.isdigit():
        length = int(length_str)
    else:
        return None

    if rng == "*":
        return ds.ContentRange(units, None, None, length, on_update=on_update)
    elif "-" not in rng:
        return None

    start_str, stop_str = rng.split("-", 1)
    try:
        start = int(start_str)
        stop = int(stop_str) + 1
    except ValueError:
        return None

    if is_byte_range_valid(start, stop, length):
        return ds.ContentRange(units, start, stop, length, on_update=on_update)

    return None


def quote_etag(etag: str, weak: bool = False) -> str:
    """Quote an etag.

    :param etag: the etag to quote.
    :param weak: set to `True` to tag it "weak".
    """
    if '"' in etag:
        raise ValueError("invalid etag")
    etag = f'"{etag}"'
    if weak:
        etag = f"W/{etag}"
    return etag


def unquote_etag(
    etag: t.Optional[str],
) -> t.Union[t.Tuple[str, bool], t.Tuple[None, None]]:
    """Unquote a single etag:

    >>> unquote_etag('W/"bar"')
    ('bar', True)
    >>> unquote_etag('"bar"')
    ('bar', False)

    :param etag: the etag identifier to unquote.
    :return: a ``(etag, weak)`` tuple.
    """
    if not etag:
        return None, None
    etag = etag.strip()
    weak = False
    if etag.startswith(("W/", "w/")):
        weak = True
        etag = etag[2:]
    if etag[:1] == etag[-1:] == '"':
        etag = etag[1:-1]
    return etag, weak


def parse_etags(value: t.Optional[str]) -> "ds.ETags":
    """Parse an etag header.

    :param value: the tag header to parse
    :return: an :class:`~werkzeug.datastructures.ETags` object.
    """
    if not value:
        return ds.ETags()
    strong = []
    weak = []
    end = len(value)
    pos = 0
    while pos < end:
        match = _etag_re.match(value, pos)
        if match is None:
            break
        is_weak, quoted, raw = match.groups()
        if raw == "*":
            return ds.ETags(star_tag=True)
        elif quoted:
            raw = quoted
        if is_weak:
            weak.append(raw)
        else:
            strong.append(raw)
        pos = match.end()
    return ds.ETags(strong, weak)


def generate_etag(data: bytes) -> str:
    """Generate an etag for some data.

    .. versionchanged:: 2.0
        Use SHA-1. MD5 may not be available in some environments.
    """
    return sha1(data).hexdigest()


def parse_date(value: t.Optional[str]) -> t.Optional[datetime]:
    """Parse an :rfc:`2822` date into a timezone-aware
    :class:`datetime.datetime` object, or ``None`` if parsing fails.

    This is a wrapper for :func:`email.utils.parsedate_to_datetime`. It
    returns ``None`` if parsing fails instead of raising an exception,
    and always returns a timezone-aware datetime object. If the string
    doesn't have timezone information, it is assumed to be UTC.

    :param value: A string with a supported date format.

    .. versionchanged:: 2.0
        Return a timezone-aware datetime object. Use
        ``email.utils.parsedate_to_datetime``.
    """
    if value is None:
        return None

    try:
        dt = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)

    return dt


def cookie_date(
    expires: t.Optional[t.Union[datetime, date, int, float, struct_time]] = None
) -> str:
    """Format a datetime object or timestamp into an :rfc:`2822` date
    string for ``Set-Cookie expires``.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use :func:`http_date` instead.
    """
    warnings.warn(
        "'cookie_date' is deprecated and will be removed in Werkzeug"
        " 2.1. Use 'http_date' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return http_date(expires)


def http_date(
    timestamp: t.Optional[t.Union[datetime, date, int, float, struct_time]] = None
) -> str:
    """Format a datetime object or timestamp into an :rfc:`2822` date
    string.

    This is a wrapper for :func:`email.utils.format_datetime`. It
    assumes naive datetime objects are in UTC instead of raising an
    exception.

    :param timestamp: The datetime or timestamp to format. Defaults to
        the current time.

    .. versionchanged:: 2.0
        Use ``email.utils.format_datetime``. Accept ``date`` objects.
    """
    if isinstance(timestamp, date):
        if not isinstance(timestamp, datetime):
            # Assume plain date is midnight UTC.
            timestamp = datetime.combine(timestamp, time(), tzinfo=timezone.utc)
        else:
            # Ensure datetime is timezone-aware.
            timestamp = _dt_as_utc(timestamp)

        return email.utils.format_datetime(timestamp, usegmt=True)

    if isinstance(timestamp, struct_time):
        timestamp = mktime(timestamp)

    return email.utils.formatdate(timestamp, usegmt=True)


def parse_age(value: t.Optional[str] = None) -> t.Optional[timedelta]:
    """Parses a base-10 integer count of seconds into a timedelta.

    If parsing fails, the return value is `None`.

    :param value: a string consisting of an integer represented in base-10
    :return: a :class:`datetime.timedelta` object or `None`.
    """
    if not value:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    if seconds < 0:
        return None
    try:
        return timedelta(seconds=seconds)
    except OverflowError:
        return None


def dump_age(age: t.Optional[t.Union[timedelta, int]] = None) -> t.Optional[str]:
    """Formats the duration as a base-10 integer.

    :param age: should be an integer number of seconds,
                a :class:`datetime.timedelta` object, or,
                if the age is unknown, `None` (default).
    """
    if age is None:
        return None
    if isinstance(age, timedelta):
        age = int(age.total_seconds())
    else:
        age = int(age)

    if age < 0:
        raise ValueError("age cannot be negative")

    return str(age)


def is_resource_modified(
    environ: "WSGIEnvironment",
    etag: t.Optional[str] = None,
    data: t.Optional[bytes] = None,
    last_modified: t.Optional[t.Union[datetime, str]] = None,
    ignore_if_range: bool = True,
) -> bool:
    """Convenience method for conditional requests.

    :param environ: the WSGI environment of the request to be checked.
    :param etag: the etag for the response for comparison.
    :param data: or alternatively the data of the response to automatically
                 generate an etag using :func:`generate_etag`.
    :param last_modified: an optional date of the last modification.
    :param ignore_if_range: If `False`, `If-Range` header will be taken into
                            account.
    :return: `True` if the resource was modified, otherwise `False`.

    .. versionchanged:: 2.0
        SHA-1 is used to generate an etag value for the data. MD5 may
        not be available in some environments.

    .. versionchanged:: 1.0.0
        The check is run for methods other than ``GET`` and ``HEAD``.
    """
    if etag is None and data is not None:
        etag = generate_etag(data)
    elif data is not None:
        raise TypeError("both data and etag given")

    unmodified = False
    if isinstance(last_modified, str):
        last_modified = parse_date(last_modified)

    # HTTP doesn't use microsecond, remove it to avoid false positive
    # comparisons. Mark naive datetimes as UTC.
    if last_modified is not None:
        last_modified = _dt_as_utc(last_modified.replace(microsecond=0))

    if_range = None
    if not ignore_if_range and "HTTP_RANGE" in environ:
        # https://tools.ietf.org/html/rfc7233#section-3.2
        # A server MUST ignore an If-Range header field received in a request
        # that does not contain a Range header field.
        if_range = parse_if_range_header(environ.get("HTTP_IF_RANGE"))

    if if_range is not None and if_range.date is not None:
        modified_since: t.Optional[datetime] = if_range.date
    else:
        modified_since = parse_date(environ.get("HTTP_IF_MODIFIED_SINCE"))

    if modified_since and last_modified and last_modified <= modified_since:
        unmodified = True

    if etag:
        etag, _ = unquote_etag(etag)
        etag = t.cast(str, etag)

        if if_range is not None and if_range.etag is not None:
            unmodified = parse_etags(if_range.etag).contains(etag)
        else:
            if_none_match = parse_etags(environ.get("HTTP_IF_NONE_MATCH"))
            if if_none_match:
                # https://tools.ietf.org/html/rfc7232#section-3.2
                # "A recipient MUST use the weak comparison function when comparing
                # entity-tags for If-None-Match"
                unmodified = if_none_match.contains_weak(etag)

            # https://tools.ietf.org/html/rfc7232#section-3.1
            # "Origin server MUST use the strong comparison function when
            # comparing entity-tags for If-Match"
            if_match = parse_etags(environ.get("HTTP_IF_MATCH"))
            if if_match:
                unmodified = not if_match.is_strong(etag)

    return not unmodified


def remove_entity_headers(
    headers: t.Union["ds.Headers", t.List[t.Tuple[str, str]]],
    allowed: t.Iterable[str] = ("expires", "content-location"),
) -> None:
    """Remove all entity headers from a list or :class:`Headers` object.  This
    operation works in-place.  `Expires` and `Content-Location` headers are
    by default not removed.  The reason for this is :rfc:`2616` section
    10.3.5 which specifies some entity headers that should be sent.

    .. versionchanged:: 0.5
       added `allowed` parameter.

    :param headers: a list or :class:`Headers` object.
    :param allowed: a list of headers that should still be allowed even though
                    they are entity headers.
    """
    allowed = {x.lower() for x in allowed}
    headers[:] = [
        (key, value)
        for key, value in headers
        if not is_entity_header(key) or key.lower() in allowed
    ]


def remove_hop_by_hop_headers(
    headers: t.Union["ds.Headers", t.List[t.Tuple[str, str]]]
) -> None:
    """Remove all HTTP/1.1 "Hop-by-Hop" headers from a list or
    :class:`Headers` object.  This operation works in-place.

    .. versionadded:: 0.5

    :param headers: a list or :class:`Headers` object.
    """
    headers[:] = [
        (key, value) for key, value in headers if not is_hop_by_hop_header(key)
    ]


def is_entity_header(header: str) -> bool:
    """Check if a header is an entity header.

    .. versionadded:: 0.5

    :param header: the header to test.
    :return: `True` if it's an entity header, `False` otherwise.
    """
    return header.lower() in _entity_headers


def is_hop_by_hop_header(header: str) -> bool:
    """Check if a header is an HTTP/1.1 "Hop-by-Hop" header.

    .. versionadded:: 0.5

    :param header: the header to test.
    :return: `True` if it's an HTTP/1.1 "Hop-by-Hop" header, `False` otherwise.
    """
    return header.lower() in _hop_by_hop_headers


def parse_cookie(
    header: t.Union["WSGIEnvironment", str, bytes, None],
    charset: str = "utf-8",
    errors: str = "replace",
    cls: t.Optional[t.Type["ds.MultiDict"]] = None,
) -> "ds.MultiDict[str, str]":
    """Parse a cookie from a string or WSGI environ.

    The same key can be provided multiple times, the values are stored
    in-order. The default :class:`MultiDict` will have the first value
    first, and all values can be retrieved with
    :meth:`MultiDict.getlist`.

    :param header: The cookie header as a string, or a WSGI environ dict
        with a ``HTTP_COOKIE`` key.
    :param charset: The charset for the cookie values.
    :param errors: The error behavior for the charset decoding.
    :param cls: A dict-like class to store the parsed cookies in.
        Defaults to :class:`MultiDict`.

    .. versionchanged:: 1.0.0
        Returns a :class:`MultiDict` instead of a
        ``TypeConversionDict``.

    .. versionchanged:: 0.5
       Returns a :class:`TypeConversionDict` instead of a regular dict.
       The ``cls`` parameter was added.
    """
    if isinstance(header, dict):
        header = header.get("HTTP_COOKIE", "")
    elif header is None:
        header = ""

    # PEP 3333 sends headers through the environ as latin1 decoded
    # strings. Encode strings back to bytes for parsing.
    if isinstance(header, str):
        header = header.encode("latin1", "replace")

    if cls is None:
        cls = ds.MultiDict

    def _parse_pairs() -> t.Iterator[t.Tuple[str, str]]:
        for key, val in _cookie_parse_impl(header):  # type: ignore
            key_str = _to_str(key, charset, errors, allow_none_charset=True)

            if not key_str:
                continue

            val_str = _to_str(val, charset, errors, allow_none_charset=True)
            yield key_str, val_str

    return cls(_parse_pairs())


def dump_cookie(
    key: str,
    value: t.Union[bytes, str] = "",
    max_age: t.Optional[t.Union[timedelta, int]] = None,
    expires: t.Optional[t.Union[str, datetime, int, float]] = None,
    path: t.Optional[str] = "/",
    domain: t.Optional[str] = None,
    secure: bool = False,
    httponly: bool = False,
    charset: str = "utf-8",
    sync_expires: bool = True,
    max_size: int = 4093,
    samesite: t.Optional[str] = None,
) -> str:
    """Create a Set-Cookie header without the ``Set-Cookie`` prefix.

    The return value is usually restricted to ascii as the vast majority
    of values are properly escaped, but that is no guarantee. It's
    tunneled through latin1 as required by :pep:`3333`.

    The return value is not ASCII safe if the key contains unicode
    characters.  This is technically against the specification but
    happens in the wild.  It's strongly recommended to not use
    non-ASCII values for the keys.

    :param max_age: should be a number of seconds, or `None` (default) if
                    the cookie should last only as long as the client's
                    browser session.  Additionally `timedelta` objects
                    are accepted, too.
    :param expires: should be a `datetime` object or unix timestamp.
    :param path: limits the cookie to a given path, per default it will
                 span the whole domain.
    :param domain: Use this if you want to set a cross-domain cookie. For
                   example, ``domain=".example.com"`` will set a cookie
                   that is readable by the domain ``www.example.com``,
                   ``foo.example.com`` etc. Otherwise, a cookie will only
                   be readable by the domain that set it.
    :param secure: The cookie will only be available via HTTPS
    :param httponly: disallow JavaScript to access the cookie.  This is an
                     extension to the cookie standard and probably not
                     supported by all browsers.
    :param charset: the encoding for string values.
    :param sync_expires: automatically set expires if max_age is defined
                         but expires not.
    :param max_size: Warn if the final header value exceeds this size. The
        default, 4093, should be safely `supported by most browsers
        <cookie_>`_. Set to 0 to disable this check.
    :param samesite: Limits the scope of the cookie such that it will
        only be attached to requests if those requests are same-site.

    .. _`cookie`: http://browsercookielimits.squawky.net/

    .. versionchanged:: 1.0.0
        The string ``'None'`` is accepted for ``samesite``.
    """
    key = _to_bytes(key, charset)
    value = _to_bytes(value, charset)

    if path is not None:
        from .urls import iri_to_uri

        path = iri_to_uri(path, charset)

    domain = _make_cookie_domain(domain)

    if isinstance(max_age, timedelta):
        max_age = int(max_age.total_seconds())

    if expires is not None:
        if not isinstance(expires, str):
            expires = http_date(expires)
    elif max_age is not None and sync_expires:
        expires = http_date(datetime.now(tz=timezone.utc).timestamp() + max_age)

    if samesite is not None:
        samesite = samesite.title()

        if samesite not in {"Strict", "Lax", "None"}:
            raise ValueError("SameSite must be 'Strict', 'Lax', or 'None'.")

    buf = [key + b"=" + _cookie_quote(value)]

    # XXX: In theory all of these parameters that are not marked with `None`
    # should be quoted.  Because stdlib did not quote it before I did not
    # want to introduce quoting there now.
    for k, v, q in (
        (b"Domain", domain, True),
        (b"Expires", expires, False),
        (b"Max-Age", max_age, False),
        (b"Secure", secure, None),
        (b"HttpOnly", httponly, None),
        (b"Path", path, False),
        (b"SameSite", samesite, False),
    ):
        if q is None:
            if v:
                buf.append(k)
            continue

        if v is None:
            continue

        tmp = bytearray(k)
        if not isinstance(v, (bytes, bytearray)):
            v = _to_bytes(str(v), charset)
        if q:
            v = _cookie_quote(v)
        tmp += b"=" + v
        buf.append(bytes(tmp))

    # The return value will be an incorrectly encoded latin1 header for
    # consistency with the headers object.
    rv = b"; ".join(buf)
    rv = rv.decode("latin1")

    # Warn if the final value of the cookie is larger than the limit. If the
    # cookie is too large, then it may be silently ignored by the browser,
    # which can be quite hard to debug.
    cookie_size = len(rv)

    if max_size and cookie_size > max_size:
        value_size = len(value)
        warnings.warn(
            f"The {key.decode(charset)!r} cookie is too large: the value was"
            f" {value_size} bytes but the"
            f" header required {cookie_size - value_size} extra bytes. The final size"
            f" was {cookie_size} bytes but the limit is {max_size} bytes. Browsers may"
            f" silently ignore cookies larger than this.",
            stacklevel=2,
        )

    return rv


def is_byte_range_valid(
    start: t.Optional[int], stop: t.Optional[int], length: t.Optional[int]
) -> bool:
    """Checks if a given byte content range is valid for the given length.

    .. versionadded:: 0.7
    """
    if (start is None) != (stop is None):
        return False
    elif start is None:
        return length is None or length >= 0
    elif length is None:
        return 0 <= start < stop  # type: ignore
    elif start >= stop:  # type: ignore
        return False
    return 0 <= start < length


# circular dependencies
from . import datastructures as ds
