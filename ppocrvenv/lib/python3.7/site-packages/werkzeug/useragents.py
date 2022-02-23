import re
import typing as t
import warnings

from .user_agent import UserAgent as _BaseUserAgent

if t.TYPE_CHECKING:
    from _typeshed.wsgi import WSGIEnvironment


class _UserAgentParser:
    platform_rules: t.ClassVar[t.Iterable[t.Tuple[str, str]]] = (
        (" cros ", "chromeos"),
        ("iphone|ios", "iphone"),
        ("ipad", "ipad"),
        (r"darwin\b|mac\b|os\s*x", "macos"),
        ("win", "windows"),
        (r"android", "android"),
        ("netbsd", "netbsd"),
        ("openbsd", "openbsd"),
        ("freebsd", "freebsd"),
        ("dragonfly", "dragonflybsd"),
        ("(sun|i86)os", "solaris"),
        (r"x11\b|lin(\b|ux)?", "linux"),
        (r"nintendo\s+wii", "wii"),
        ("irix", "irix"),
        ("hp-?ux", "hpux"),
        ("aix", "aix"),
        ("sco|unix_sv", "sco"),
        ("bsd", "bsd"),
        ("amiga", "amiga"),
        ("blackberry|playbook", "blackberry"),
        ("symbian", "symbian"),
    )
    browser_rules: t.ClassVar[t.Iterable[t.Tuple[str, str]]] = (
        ("googlebot", "google"),
        ("msnbot", "msn"),
        ("yahoo", "yahoo"),
        ("ask jeeves", "ask"),
        (r"aol|america\s+online\s+browser", "aol"),
        (r"opera|opr", "opera"),
        ("edge|edg", "edge"),
        ("chrome|crios", "chrome"),
        ("seamonkey", "seamonkey"),
        ("firefox|firebird|phoenix|iceweasel", "firefox"),
        ("galeon", "galeon"),
        ("safari|version", "safari"),
        ("webkit", "webkit"),
        ("camino", "camino"),
        ("konqueror", "konqueror"),
        ("k-meleon", "kmeleon"),
        ("netscape", "netscape"),
        (r"msie|microsoft\s+internet\s+explorer|trident/.+? rv:", "msie"),
        ("lynx", "lynx"),
        ("links", "links"),
        ("Baiduspider", "baidu"),
        ("bingbot", "bing"),
        ("mozilla", "mozilla"),
    )

    _browser_version_re = r"(?:{pattern})[/\sa-z(]*(\d+[.\da-z]+)?"
    _language_re = re.compile(
        r"(?:;\s*|\s+)(\b\w{2}\b(?:-\b\w{2}\b)?)\s*;|"
        r"(?:\(|\[|;)\s*(\b\w{2}\b(?:-\b\w{2}\b)?)\s*(?:\]|\)|;)"
    )

    def __init__(self) -> None:
        self.platforms = [(b, re.compile(a, re.I)) for a, b in self.platform_rules]
        self.browsers = [
            (b, re.compile(self._browser_version_re.format(pattern=a), re.I))
            for a, b in self.browser_rules
        ]

    def __call__(
        self, user_agent: str
    ) -> t.Tuple[t.Optional[str], t.Optional[str], t.Optional[str], t.Optional[str]]:
        platform: t.Optional[str]
        browser: t.Optional[str]
        version: t.Optional[str]
        language: t.Optional[str]

        for platform, regex in self.platforms:  # noqa: B007
            match = regex.search(user_agent)
            if match is not None:
                break
        else:
            platform = None

        # Except for Trident, all browser key words come after the last ')'
        last_closing_paren = 0
        if (
            not re.compile(r"trident/.+? rv:", re.I).search(user_agent)
            and ")" in user_agent
            and user_agent[-1] != ")"
        ):
            last_closing_paren = user_agent.rindex(")")

        for browser, regex in self.browsers:  # noqa: B007
            match = regex.search(user_agent[last_closing_paren:])
            if match is not None:
                version = match.group(1)
                break
        else:
            browser = version = None
        match = self._language_re.search(user_agent)
        if match is not None:
            language = match.group(1) or match.group(2)
        else:
            language = None
        return platform, browser, version, language


# It wasn't public, but users might have imported it anyway, show a
# warning if a user created an instance.
class UserAgentParser(_UserAgentParser):
    """A simple user agent parser.  Used by the `UserAgent`.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Use a dedicated parser library
        instead.
    """

    def __init__(self) -> None:
        warnings.warn(
            "'UserAgentParser' is deprecated and will be removed in"
            " Werkzeug 2.1. Use a dedicated parser library instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()


class _deprecated_property(property):
    def __init__(self, fget: t.Callable[["_UserAgent"], t.Any]) -> None:
        super().__init__(fget)
        self.message = (
            "The built-in user agent parser is deprecated and will be"
            f" removed in Werkzeug 2.1. The {fget.__name__!r} property"
            " will be 'None'. Subclass 'werkzeug.user_agent.UserAgent'"
            " and set 'Request.user_agent_class' to use a different"
            " parser."
        )

    def __get__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        warnings.warn(self.message, DeprecationWarning, stacklevel=3)
        return super().__get__(*args, **kwargs)


# This is what Request.user_agent returns for now, only show warnings on
# attribute access, not creation.
class _UserAgent(_BaseUserAgent):
    _parser = _UserAgentParser()

    def __init__(self, string: str) -> None:
        super().__init__(string)
        info = self._parser(string)
        self._platform, self._browser, self._version, self._language = info

    @_deprecated_property
    def platform(self) -> t.Optional[str]:  # type: ignore
        return self._platform

    @_deprecated_property
    def browser(self) -> t.Optional[str]:  # type: ignore
        return self._browser

    @_deprecated_property
    def version(self) -> t.Optional[str]:  # type: ignore
        return self._version

    @_deprecated_property
    def language(self) -> t.Optional[str]:  # type: ignore
        return self._language


# This is what users might be importing, show warnings on create.
class UserAgent(_UserAgent):
    """Represents a parsed user agent header value.

    This uses a basic parser to try to extract some information from the
    header.

    :param environ_or_string: The header value to parse, or a WSGI
        environ containing the header.

    .. deprecated:: 2.0
        Will be removed in Werkzeug 2.1. Subclass
        :class:`werkzeug.user_agent.UserAgent` (note the new module
        name) to use a dedicated parser instead.

    .. versionchanged:: 2.0
        Passing a WSGI environ is deprecated and will be removed in 2.1.
    """

    def __init__(self, environ_or_string: "t.Union[str, WSGIEnvironment]") -> None:
        if isinstance(environ_or_string, dict):
            warnings.warn(
                "Passing an environ to 'UserAgent' is deprecated and"
                " will be removed in Werkzeug 2.1. Pass the header"
                " value string instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            string = environ_or_string.get("HTTP_USER_AGENT", "")
        else:
            string = environ_or_string

        warnings.warn(
            "The 'werkzeug.useragents' module is deprecated and will be"
            " removed in Werkzeug 2.1. The new base API is"
            " 'werkzeug.user_agent.UserAgent'.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(string)
