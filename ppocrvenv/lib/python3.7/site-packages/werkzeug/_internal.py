import inspect
import logging
import operator
import re
import string
import sys
import typing
import typing as t
from datetime import date
from datetime import datetime
from datetime import timezone
from itertools import chain
from weakref import WeakKeyDictionary

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment
    from .wrappers.request import Request  # noqa: F401

_logger: t.Optional[logging.Logger] = None
_signature_cache = WeakKeyDictionary()  # type: ignore
_epoch_ord = date(1970, 1, 1).toordinal()
_legal_cookie_chars = frozenset(
    c.encode("ascii")
    for c in f"{string.ascii_letters}{string.digits}/=!#$%&'*+-.^_`|~:"
)

_cookie_quoting_map = {b",": b"\\054", b";": b"\\073", b'"': b'\\"', b"\\": b"\\\\"}
for _i in chain(range(32), range(127, 256)):
    _cookie_quoting_map[_i.to_bytes(1, sys.byteorder)] = f"\\{_i:03o}".encode("latin1")

_octal_re = re.compile(br"\\[0-3][0-7][0-7]")
_quote_re = re.compile(br"[\\].")
_legal_cookie_chars_re = br"[\w\d!#%&\'~_`><@,:/\$\*\+\-\.\^\|\)\(\?\}\{\=]"
_cookie_re = re.compile(
    br"""
    (?P<key>[^=;]+)
    (?:\s*=\s*
        (?P<val>
            "(?:[^\\"]|\\.)*" |
             (?:.*?)
        )
    )?
    \s*;
""",
    flags=re.VERBOSE,
)


class _Missing:
    def __repr__(self) -> str:
        return "no value"

    def __reduce__(self) -> str:
        return "_missing"


_missing = _Missing()


@typing.overload
def _make_encode_wrapper(reference: str) -> t.Callable[[str], str]:
    ...


@typing.overload
def _make_encode_wrapper(reference: bytes) -> t.Callable[[str], bytes]:
    ...


def _make_encode_wrapper(reference: t.AnyStr) -> t.Callable[[str], t.AnyStr]:
    """Create a function that will be called with a string argument. If
    the reference is bytes, values will be encoded to bytes.
    """
    if isinstance(reference, str):
        return lambda x: x

    return operator.methodcaller("encode", "latin1")


def _check_str_tuple(value: t.Tuple[t.AnyStr, ...]) -> None:
    """Ensure tuple items are all strings or all bytes."""
    if not value:
        return

    item_type = str if isinstance(value[0], str) else bytes

    if any(not isinstance(item, item_type) for item in value):
        raise TypeError(f"Cannot mix str and bytes arguments (got {value!r})")


_default_encoding = sys.getdefaultencoding()


def _to_bytes(
    x: t.Union[str, bytes], charset: str = _default_encoding, errors: str = "strict"
) -> bytes:
    if x is None or isinstance(x, bytes):
        return x

    if isinstance(x, (bytearray, memoryview)):
        return bytes(x)

    if isinstance(x, str):
        return x.encode(charset, errors)

    raise TypeError("Expected bytes")


@typing.overload
def _to_str(  # type: ignore
    x: None,
    charset: t.Optional[str] = ...,
    errors: str = ...,
    allow_none_charset: bool = ...,
) -> None:
    ...


@typing.overload
def _to_str(
    x: t.Any,
    charset: t.Optional[str] = ...,
    errors: str = ...,
    allow_none_charset: bool = ...,
) -> str:
    ...


def _to_str(
    x: t.Optional[t.Any],
    charset: t.Optional[str] = _default_encoding,
    errors: str = "strict",
    allow_none_charset: bool = False,
) -> t.Optional[t.Union[str, bytes]]:
    if x is None or isinstance(x, str):
        return x

    if not isinstance(x, (bytes, bytearray)):
        return str(x)

    if charset is None:
        if allow_none_charset:
            return x

    return x.decode(charset, errors)  # type: ignore


def _wsgi_decoding_dance(
    s: str, charset: str = "utf-8", errors: str = "replace"
) -> str:
    return s.encode("latin1").decode(charset, errors)


def _wsgi_encoding_dance(
    s: str, charset: str = "utf-8", errors: str = "replace"
) -> str:
    if isinstance(s, bytes):
        return s.decode("latin1", errors)

    return s.encode(charset).decode("latin1", errors)


def _get_environ(obj: t.Union["WSGIEnvironment", "Request"]) -> "WSGIEnvironment":
    env = getattr(obj, "environ", obj)
    assert isinstance(
        env, dict
    ), f"{type(obj).__name__!r} is not a WSGI environment (has to be a dict)"
    return env


def _has_level_handler(logger: logging.Logger) -> bool:
    """Check if there is a handler in the logging chain that will handle
    the given logger's effective level.
    """
    level = logger.getEffectiveLevel()
    current = logger

    while current:
        if any(handler.level <= level for handler in current.handlers):
            return True

        if not current.propagate:
            break

        current = current.parent  # type: ignore

    return False


class _ColorStreamHandler(logging.StreamHandler):
    """On Windows, wrap stream with Colorama for ANSI style support."""

    def __init__(self) -> None:
        try:
            import colorama
        except ImportError:
            stream = None
        else:
            stream = colorama.AnsiToWin32(sys.stderr)

        super().__init__(stream)


def _log(type: str, message: str, *args: t.Any, **kwargs: t.Any) -> None:
    """Log a message to the 'werkzeug' logger.

    The logger is created the first time it is needed. If there is no
    level set, it is set to :data:`logging.INFO`. If there is no handler
    for the logger's effective level, a :class:`logging.StreamHandler`
    is added.
    """
    global _logger

    if _logger is None:
        _logger = logging.getLogger("werkzeug")

        if _logger.level == logging.NOTSET:
            _logger.setLevel(logging.INFO)

        if not _has_level_handler(_logger):
            _logger.addHandler(_ColorStreamHandler())

    getattr(_logger, type)(message.rstrip(), *args, **kwargs)


def _parse_signature(func):  # type: ignore
    """Return a signature object for the function.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1 along with ``utils.bind`` and
        ``validate_arguments``.
    """
    # if we have a cached validator for this function, return it
    parse = _signature_cache.get(func)
    if parse is not None:
        return parse

    # inspect the function signature and collect all the information
    tup = inspect.getfullargspec(func)
    positional, vararg_var, kwarg_var, defaults = tup[:4]
    defaults = defaults or ()
    arg_count = len(positional)
    arguments = []
    for idx, name in enumerate(positional):
        if isinstance(name, list):
            raise TypeError(
                "cannot parse functions that unpack tuples in the function signature"
            )
        try:
            default = defaults[idx - arg_count]
        except IndexError:
            param = (name, False, None)
        else:
            param = (name, True, default)
        arguments.append(param)
    arguments = tuple(arguments)

    def parse(args, kwargs):  # type: ignore
        new_args = []
        missing = []
        extra = {}

        # consume as many arguments as positional as possible
        for idx, (name, has_default, default) in enumerate(arguments):
            try:
                new_args.append(args[idx])
            except IndexError:
                try:
                    new_args.append(kwargs.pop(name))
                except KeyError:
                    if has_default:
                        new_args.append(default)
                    else:
                        missing.append(name)
            else:
                if name in kwargs:
                    extra[name] = kwargs.pop(name)

        # handle extra arguments
        extra_positional = args[arg_count:]
        if vararg_var is not None:
            new_args.extend(extra_positional)
            extra_positional = ()
        if kwargs and kwarg_var is None:
            extra.update(kwargs)
            kwargs = {}

        return (
            new_args,
            kwargs,
            missing,
            extra,
            extra_positional,
            arguments,
            vararg_var,
            kwarg_var,
        )

    _signature_cache[func] = parse
    return parse


@typing.overload
def _dt_as_utc(dt: None) -> None:
    ...


@typing.overload
def _dt_as_utc(dt: datetime) -> datetime:
    ...


def _dt_as_utc(dt: t.Optional[datetime]) -> t.Optional[datetime]:
    if dt is None:
        return dt

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo != timezone.utc:
        return dt.astimezone(timezone.utc)

    return dt


_TAccessorValue = t.TypeVar("_TAccessorValue")


class _DictAccessorProperty(t.Generic[_TAccessorValue]):
    """Baseclass for `environ_property` and `header_property`."""

    read_only = False

    def __init__(
        self,
        name: str,
        default: t.Optional[_TAccessorValue] = None,
        load_func: t.Optional[t.Callable[[str], _TAccessorValue]] = None,
        dump_func: t.Optional[t.Callable[[_TAccessorValue], str]] = None,
        read_only: t.Optional[bool] = None,
        doc: t.Optional[str] = None,
    ) -> None:
        self.name = name
        self.default = default
        self.load_func = load_func
        self.dump_func = dump_func
        if read_only is not None:
            self.read_only = read_only
        self.__doc__ = doc

    def lookup(self, instance: t.Any) -> t.MutableMapping[str, t.Any]:
        raise NotImplementedError

    @typing.overload
    def __get__(
        self, instance: None, owner: type
    ) -> "_DictAccessorProperty[_TAccessorValue]":
        ...

    @typing.overload
    def __get__(self, instance: t.Any, owner: type) -> _TAccessorValue:
        ...

    def __get__(
        self, instance: t.Optional[t.Any], owner: type
    ) -> t.Union[_TAccessorValue, "_DictAccessorProperty[_TAccessorValue]"]:
        if instance is None:
            return self

        storage = self.lookup(instance)

        if self.name not in storage:
            return self.default  # type: ignore

        value = storage[self.name]

        if self.load_func is not None:
            try:
                return self.load_func(value)
            except (ValueError, TypeError):
                return self.default  # type: ignore

        return value  # type: ignore

    def __set__(self, instance: t.Any, value: _TAccessorValue) -> None:
        if self.read_only:
            raise AttributeError("read only property")

        if self.dump_func is not None:
            self.lookup(instance)[self.name] = self.dump_func(value)
        else:
            self.lookup(instance)[self.name] = value

    def __delete__(self, instance: t.Any) -> None:
        if self.read_only:
            raise AttributeError("read only property")

        self.lookup(instance).pop(self.name, None)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def _cookie_quote(b: bytes) -> bytes:
    buf = bytearray()
    all_legal = True
    _lookup = _cookie_quoting_map.get
    _push = buf.extend

    for char_int in b:
        char = char_int.to_bytes(1, sys.byteorder)
        if char not in _legal_cookie_chars:
            all_legal = False
            char = _lookup(char, char)
        _push(char)

    if all_legal:
        return bytes(buf)
    return bytes(b'"' + buf + b'"')


def _cookie_unquote(b: bytes) -> bytes:
    if len(b) < 2:
        return b
    if b[:1] != b'"' or b[-1:] != b'"':
        return b

    b = b[1:-1]

    i = 0
    n = len(b)
    rv = bytearray()
    _push = rv.extend

    while 0 <= i < n:
        o_match = _octal_re.search(b, i)
        q_match = _quote_re.search(b, i)
        if not o_match and not q_match:
            rv.extend(b[i:])
            break
        j = k = -1
        if o_match:
            j = o_match.start(0)
        if q_match:
            k = q_match.start(0)
        if q_match and (not o_match or k < j):
            _push(b[i:k])
            _push(b[k + 1 : k + 2])
            i = k + 2
        else:
            _push(b[i:j])
            rv.append(int(b[j + 1 : j + 4], 8))
            i = j + 4

    return bytes(rv)


def _cookie_parse_impl(b: bytes) -> t.Iterator[t.Tuple[bytes, bytes]]:
    """Lowlevel cookie parsing facility that operates on bytes."""
    i = 0
    n = len(b)

    while i < n:
        match = _cookie_re.search(b + b";", i)
        if not match:
            break

        key = match.group("key").strip()
        value = match.group("val") or b""
        i = match.end(0)

        yield key, _cookie_unquote(value)


def _encode_idna(domain: str) -> bytes:
    # If we're given bytes, make sure they fit into ASCII
    if isinstance(domain, bytes):
        domain.decode("ascii")
        return domain

    # Otherwise check if it's already ascii, then return
    try:
        return domain.encode("ascii")
    except UnicodeError:
        pass

    # Otherwise encode each part separately
    return b".".join(p.encode("idna") for p in domain.split("."))


def _decode_idna(domain: t.Union[str, bytes]) -> str:
    # If the input is a string try to encode it to ascii to do the idna
    # decoding. If that fails because of a unicode error, then we
    # already have a decoded idna domain.
    if isinstance(domain, str):
        try:
            domain = domain.encode("ascii")
        except UnicodeError:
            return domain  # type: ignore

    # Decode each part separately. If a part fails, try to decode it
    # with ascii and silently ignore errors. This makes sense because
    # the idna codec does not have error handling.
    def decode_part(part: bytes) -> str:
        try:
            return part.decode("idna")
        except UnicodeError:
            return part.decode("ascii", "ignore")

    return ".".join(decode_part(p) for p in domain.split(b"."))


@typing.overload
def _make_cookie_domain(domain: None) -> None:
    ...


@typing.overload
def _make_cookie_domain(domain: str) -> bytes:
    ...


def _make_cookie_domain(domain: t.Optional[str]) -> t.Optional[bytes]:
    if domain is None:
        return None
    domain = _encode_idna(domain)
    if b":" in domain:
        domain = domain.split(b":", 1)[0]
    if b"." in domain:
        return domain
    raise ValueError(
        "Setting 'domain' for a cookie on a server running locally (ex: "
        "localhost) is not supported by complying browsers. You should "
        "have something like: '127.0.0.1 localhost dev.localhost' on "
        "your hosts file and then point your server to run on "
        "'dev.localhost' and also set 'domain' for 'dev.localhost'"
    )


def _easteregg(app: t.Optional["WSGIApplication"] = None) -> "WSGIApplication":
    """Like the name says.  But who knows how it works?"""

    def bzzzzzzz(gyver: bytes) -> str:
        import base64
        import zlib

        return zlib.decompress(base64.b64decode(gyver)).decode("ascii")

    gyver = "\n".join(
        [
            x + (77 - len(x)) * " "
            for x in bzzzzzzz(
                b"""
eJyFlzuOJDkMRP06xRjymKgDJCDQStBYT8BCgK4gTwfQ2fcFs2a2FzvZk+hvlcRvRJD148efHt9m
9Xz94dRY5hGt1nrYcXx7us9qlcP9HHNh28rz8dZj+q4rynVFFPdlY4zH873NKCexrDM6zxxRymzz
4QIxzK4bth1PV7+uHn6WXZ5C4ka/+prFzx3zWLMHAVZb8RRUxtFXI5DTQ2n3Hi2sNI+HK43AOWSY
jmEzE4naFp58PdzhPMdslLVWHTGUVpSxImw+pS/D+JhzLfdS1j7PzUMxij+mc2U0I9zcbZ/HcZxc
q1QjvvcThMYFnp93agEx392ZdLJWXbi/Ca4Oivl4h/Y1ErEqP+lrg7Xa4qnUKu5UE9UUA4xeqLJ5
jWlPKJvR2yhRI7xFPdzPuc6adXu6ovwXwRPXXnZHxlPtkSkqWHilsOrGrvcVWXgGP3daXomCj317
8P2UOw/NnA0OOikZyFf3zZ76eN9QXNwYdD8f8/LdBRFg0BO3bB+Pe/+G8er8tDJv83XTkj7WeMBJ
v/rnAfdO51d6sFglfi8U7zbnr0u9tyJHhFZNXYfH8Iafv2Oa+DT6l8u9UYlajV/hcEgk1x8E8L/r
XJXl2SK+GJCxtnyhVKv6GFCEB1OO3f9YWAIEbwcRWv/6RPpsEzOkXURMN37J0PoCSYeBnJQd9Giu
LxYQJNlYPSo/iTQwgaihbART7Fcyem2tTSCcwNCs85MOOpJtXhXDe0E7zgZJkcxWTar/zEjdIVCk
iXy87FW6j5aGZhttDBoAZ3vnmlkx4q4mMmCdLtnHkBXFMCReqthSGkQ+MDXLLCpXwBs0t+sIhsDI
tjBB8MwqYQpLygZ56rRHHpw+OAVyGgaGRHWy2QfXez+ZQQTTBkmRXdV/A9LwH6XGZpEAZU8rs4pE
1R4FQ3Uwt8RKEtRc0/CrANUoes3EzM6WYcFyskGZ6UTHJWenBDS7h163Eo2bpzqxNE9aVgEM2CqI
GAJe9Yra4P5qKmta27VjzYdR04Vc7KHeY4vs61C0nbywFmcSXYjzBHdiEjraS7PGG2jHHTpJUMxN
Jlxr3pUuFvlBWLJGE3GcA1/1xxLcHmlO+LAXbhrXah1tD6Ze+uqFGdZa5FM+3eHcKNaEarutAQ0A
QMAZHV+ve6LxAwWnXbbSXEG2DmCX5ijeLCKj5lhVFBrMm+ryOttCAeFpUdZyQLAQkA06RLs56rzG
8MID55vqr/g64Qr/wqwlE0TVxgoiZhHrbY2h1iuuyUVg1nlkpDrQ7Vm1xIkI5XRKLedN9EjzVchu
jQhXcVkjVdgP2O99QShpdvXWoSwkp5uMwyjt3jiWCqWGSiaaPAzohjPanXVLbM3x0dNskJsaCEyz
DTKIs+7WKJD4ZcJGfMhLFBf6hlbnNkLEePF8Cx2o2kwmYF4+MzAxa6i+6xIQkswOqGO+3x9NaZX8
MrZRaFZpLeVTYI9F/djY6DDVVs340nZGmwrDqTCiiqD5luj3OzwpmQCiQhdRYowUYEA3i1WWGwL4
GCtSoO4XbIPFeKGU13XPkDf5IdimLpAvi2kVDVQbzOOa4KAXMFlpi/hV8F6IDe0Y2reg3PuNKT3i
RYhZqtkQZqSB2Qm0SGtjAw7RDwaM1roESC8HWiPxkoOy0lLTRFG39kvbLZbU9gFKFRvixDZBJmpi
Xyq3RE5lW00EJjaqwp/v3EByMSpVZYsEIJ4APaHmVtpGSieV5CALOtNUAzTBiw81GLgC0quyzf6c
NlWknzJeCsJ5fup2R4d8CYGN77mu5vnO1UqbfElZ9E6cR6zbHjgsr9ly18fXjZoPeDjPuzlWbFwS
pdvPkhntFvkc13qb9094LL5NrA3NIq3r9eNnop9DizWOqCEbyRBFJTHn6Tt3CG1o8a4HevYh0XiJ
sR0AVVHuGuMOIfbuQ/OKBkGRC6NJ4u7sbPX8bG/n5sNIOQ6/Y/BX3IwRlTSabtZpYLB85lYtkkgm
p1qXK3Du2mnr5INXmT/78KI12n11EFBkJHHp0wJyLe9MvPNUGYsf+170maayRoy2lURGHAIapSpQ
krEDuNoJCHNlZYhKpvw4mspVWxqo415n8cD62N9+EfHrAvqQnINStetek7RY2Urv8nxsnGaZfRr/
nhXbJ6m/yl1LzYqscDZA9QHLNbdaSTTr+kFg3bC0iYbX/eQy0Bv3h4B50/SGYzKAXkCeOLI3bcAt
mj2Z/FM1vQWgDynsRwNvrWnJHlespkrp8+vO1jNaibm+PhqXPPv30YwDZ6jApe3wUjFQobghvW9p
7f2zLkGNv8b191cD/3vs9Q833z8t"""
            ).splitlines()
        ]
    )

    def easteregged(
        environ: "WSGIEnvironment", start_response: "StartResponse"
    ) -> t.Iterable[bytes]:
        def injecting_start_response(
            status: str, headers: t.List[t.Tuple[str, str]], exc_info: t.Any = None
        ) -> t.Callable[[bytes], t.Any]:
            headers.append(("X-Powered-By", "Werkzeug"))
            return start_response(status, headers, exc_info)

        if app is not None and environ.get("QUERY_STRING") != "macgybarchakku":
            return app(environ, injecting_start_response)
        injecting_start_response("200 OK", [("Content-Type", "text/html")])
        return [
            f"""\
<!DOCTYPE html>
<html>
<head>
<title>About Werkzeug</title>
<style type="text/css">
  body {{ font: 15px Georgia, serif; text-align: center; }}
  a {{ color: #333; text-decoration: none; }}
  h1 {{ font-size: 30px; margin: 20px 0 10px 0; }}
  p {{ margin: 0 0 30px 0; }}
  pre {{ font: 11px 'Consolas', 'Monaco', monospace; line-height: 0.95; }}
</style>
</head>
<body>
<h1><a href="http://werkzeug.pocoo.org/">Werkzeug</a></h1>
<p>the Swiss Army knife of Python web development.</p>
<pre>{gyver}\n\n\n</pre>
</body>
</html>""".encode(
                "latin1"
            )
        ]

    return easteregged
