"""When it comes to combining multiple controller or view functions
(however you want to call them) you need a dispatcher. A simple way
would be applying regular expression tests on the ``PATH_INFO`` and
calling registered callback functions that return the value then.

This module implements a much more powerful system than simple regular
expression matching because it can also convert values in the URLs and
build URLs.

Here a simple example that creates a URL map for an application with
two subdomains (www and kb) and some URL rules:

.. code-block:: python

    m = Map([
        # Static URLs
        Rule('/', endpoint='static/index'),
        Rule('/about', endpoint='static/about'),
        Rule('/help', endpoint='static/help'),
        # Knowledge Base
        Subdomain('kb', [
            Rule('/', endpoint='kb/index'),
            Rule('/browse/', endpoint='kb/browse'),
            Rule('/browse/<int:id>/', endpoint='kb/browse'),
            Rule('/browse/<int:id>/<int:page>', endpoint='kb/browse')
        ])
    ], default_subdomain='www')

If the application doesn't use subdomains it's perfectly fine to not set
the default subdomain and not use the `Subdomain` rule factory. The
endpoint in the rules can be anything, for example import paths or
unique identifiers. The WSGI application can use those endpoints to get the
handler for that URL.  It doesn't have to be a string at all but it's
recommended.

Now it's possible to create a URL adapter for one of the subdomains and
build URLs:

.. code-block:: python

    c = m.bind('example.com')

    c.build("kb/browse", dict(id=42))
    'http://kb.example.com/browse/42/'

    c.build("kb/browse", dict())
    'http://kb.example.com/browse/'

    c.build("kb/browse", dict(id=42, page=3))
    'http://kb.example.com/browse/42/3'

    c.build("static/about")
    '/about'

    c.build("static/index", force_external=True)
    'http://www.example.com/'

    c = m.bind('example.com', subdomain='kb')

    c.build("static/about")
    'http://www.example.com/about'

The first argument to bind is the server name *without* the subdomain.
Per default it will assume that the script is mounted on the root, but
often that's not the case so you can provide the real mount point as
second argument:

.. code-block:: python

    c = m.bind('example.com', '/applications/example')

The third argument can be the subdomain, if not given the default
subdomain is used.  For more details about binding have a look at the
documentation of the `MapAdapter`.

And here is how you can match URLs:

.. code-block:: python

    c = m.bind('example.com')

    c.match("/")
    ('static/index', {})

    c.match("/about")
    ('static/about', {})

    c = m.bind('example.com', '/', 'kb')

    c.match("/")
    ('kb/index', {})

    c.match("/browse/42/23")
    ('kb/browse', {'id': 42, 'page': 23})

If matching fails you get a ``NotFound`` exception, if the rule thinks
it's a good idea to redirect (for example because the URL was defined
to have a slash at the end but the request was missing that slash) it
will raise a ``RequestRedirect`` exception. Both are subclasses of
``HTTPException`` so you can use those errors as responses in the
application.

If matching succeeded but the URL rule was incompatible to the given
method (for example there were only rules for ``GET`` and ``HEAD`` but
routing tried to match a ``POST`` request) a ``MethodNotAllowed``
exception is raised.
"""
import ast
import difflib
import posixpath
import re
import typing
import typing as t
import uuid
import warnings
from pprint import pformat
from string import Template
from threading import Lock
from types import CodeType

from ._internal import _encode_idna
from ._internal import _get_environ
from ._internal import _to_bytes
from ._internal import _to_str
from ._internal import _wsgi_decoding_dance
from .datastructures import ImmutableDict
from .datastructures import MultiDict
from .exceptions import BadHost
from .exceptions import BadRequest
from .exceptions import HTTPException
from .exceptions import MethodNotAllowed
from .exceptions import NotFound
from .urls import _fast_url_quote
from .urls import url_encode
from .urls import url_join
from .urls import url_quote
from .urls import url_unquote
from .utils import cached_property
from .utils import redirect
from .wsgi import get_host

if t.TYPE_CHECKING:
    import typing_extensions as te
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment
    from .wrappers.response import Response

_rule_re = re.compile(
    r"""
    (?P<static>[^<]*)                           # static rule data
    <
    (?:
        (?P<converter>[a-zA-Z_][a-zA-Z0-9_]*)   # converter name
        (?:\((?P<args>.*?)\))?                  # converter arguments
        \:                                      # variable delimiter
    )?
    (?P<variable>[a-zA-Z_][a-zA-Z0-9_]*)        # variable name
    >
    """,
    re.VERBOSE,
)
_simple_rule_re = re.compile(r"<([^>]+)>")
_converter_args_re = re.compile(
    r"""
    ((?P<name>\w+)\s*=\s*)?
    (?P<value>
        True|False|
        \d+.\d+|
        \d+.|
        \d+|
        [\w\d_.]+|
        [urUR]?(?P<stringval>"[^"]*?"|'[^']*')
    )\s*,
    """,
    re.VERBOSE,
)


_PYTHON_CONSTANTS = {"None": None, "True": True, "False": False}


def _pythonize(value: str) -> t.Union[None, bool, int, float, str]:
    if value in _PYTHON_CONSTANTS:
        return _PYTHON_CONSTANTS[value]
    for convert in int, float:
        try:
            return convert(value)  # type: ignore
        except ValueError:
            pass
    if value[:1] == value[-1:] and value[0] in "\"'":
        value = value[1:-1]
    return str(value)


def parse_converter_args(argstr: str) -> t.Tuple[t.Tuple, t.Dict[str, t.Any]]:
    argstr += ","
    args = []
    kwargs = {}

    for item in _converter_args_re.finditer(argstr):
        value = item.group("stringval")
        if value is None:
            value = item.group("value")
        value = _pythonize(value)
        if not item.group("name"):
            args.append(value)
        else:
            name = item.group("name")
            kwargs[name] = value

    return tuple(args), kwargs


def parse_rule(rule: str) -> t.Iterator[t.Tuple[t.Optional[str], t.Optional[str], str]]:
    """Parse a rule and return it as generator. Each iteration yields tuples
    in the form ``(converter, arguments, variable)``. If the converter is
    `None` it's a static url part, otherwise it's a dynamic one.

    :internal:
    """
    pos = 0
    end = len(rule)
    do_match = _rule_re.match
    used_names = set()
    while pos < end:
        m = do_match(rule, pos)
        if m is None:
            break
        data = m.groupdict()
        if data["static"]:
            yield None, None, data["static"]
        variable = data["variable"]
        converter = data["converter"] or "default"
        if variable in used_names:
            raise ValueError(f"variable name {variable!r} used twice.")
        used_names.add(variable)
        yield converter, data["args"] or None, variable
        pos = m.end()
    if pos < end:
        remaining = rule[pos:]
        if ">" in remaining or "<" in remaining:
            raise ValueError(f"malformed url rule: {rule!r}")
        yield None, None, remaining


class RoutingException(Exception):
    """Special exceptions that require the application to redirect, notifying
    about missing urls, etc.

    :internal:
    """


class RequestRedirect(HTTPException, RoutingException):
    """Raise if the map requests a redirect. This is for example the case if
    `strict_slashes` are activated and an url that requires a trailing slash.

    The attribute `new_url` contains the absolute destination url.
    """

    code = 308

    def __init__(self, new_url: str) -> None:
        super().__init__(new_url)
        self.new_url = new_url

    def get_response(
        self,
        environ: t.Optional["WSGIEnvironment"] = None,
        scope: t.Optional[dict] = None,
    ) -> "Response":
        return redirect(self.new_url, self.code)


class RequestPath(RoutingException):
    """Internal exception."""

    __slots__ = ("path_info",)

    def __init__(self, path_info: str) -> None:
        super().__init__()
        self.path_info = path_info


class RequestAliasRedirect(RoutingException):  # noqa: B903
    """This rule is an alias and wants to redirect to the canonical URL."""

    def __init__(self, matched_values: t.Mapping[str, t.Any]) -> None:
        super().__init__()
        self.matched_values = matched_values


class BuildError(RoutingException, LookupError):
    """Raised if the build system cannot find a URL for an endpoint with the
    values provided.
    """

    def __init__(
        self,
        endpoint: str,
        values: t.Mapping[str, t.Any],
        method: t.Optional[str],
        adapter: t.Optional["MapAdapter"] = None,
    ) -> None:
        super().__init__(endpoint, values, method)
        self.endpoint = endpoint
        self.values = values
        self.method = method
        self.adapter = adapter

    @cached_property
    def suggested(self) -> t.Optional["Rule"]:
        return self.closest_rule(self.adapter)

    def closest_rule(self, adapter: t.Optional["MapAdapter"]) -> t.Optional["Rule"]:
        def _score_rule(rule: "Rule") -> float:
            return sum(
                [
                    0.98
                    * difflib.SequenceMatcher(
                        None, rule.endpoint, self.endpoint
                    ).ratio(),
                    0.01 * bool(set(self.values or ()).issubset(rule.arguments)),
                    0.01 * bool(rule.methods and self.method in rule.methods),
                ]
            )

        if adapter and adapter.map._rules:
            return max(adapter.map._rules, key=_score_rule)

        return None

    def __str__(self) -> str:
        message = [f"Could not build url for endpoint {self.endpoint!r}"]
        if self.method:
            message.append(f" ({self.method!r})")
        if self.values:
            message.append(f" with values {sorted(self.values)!r}")
        message.append(".")
        if self.suggested:
            if self.endpoint == self.suggested.endpoint:
                if (
                    self.method
                    and self.suggested.methods is not None
                    and self.method not in self.suggested.methods
                ):
                    message.append(
                        " Did you mean to use methods"
                        f" {sorted(self.suggested.methods)!r}?"
                    )
                missing_values = self.suggested.arguments.union(
                    set(self.suggested.defaults or ())
                ) - set(self.values.keys())
                if missing_values:
                    message.append(
                        f" Did you forget to specify values {sorted(missing_values)!r}?"
                    )
            else:
                message.append(f" Did you mean {self.suggested.endpoint!r} instead?")
        return "".join(message)


class WebsocketMismatch(BadRequest):
    """The only matched rule is either a WebSocket and the request is
    HTTP, or the rule is HTTP and the request is a WebSocket.
    """


class ValidationError(ValueError):
    """Validation error.  If a rule converter raises this exception the rule
    does not match the current URL and the next URL is tried.
    """


class RuleFactory:
    """As soon as you have more complex URL setups it's a good idea to use rule
    factories to avoid repetitive tasks.  Some of them are builtin, others can
    be added by subclassing `RuleFactory` and overriding `get_rules`.
    """

    def get_rules(self, map: "Map") -> t.Iterable["Rule"]:
        """Subclasses of `RuleFactory` have to override this method and return
        an iterable of rules."""
        raise NotImplementedError()


class Subdomain(RuleFactory):
    """All URLs provided by this factory have the subdomain set to a
    specific domain. For example if you want to use the subdomain for
    the current language this can be a good setup::

        url_map = Map([
            Rule('/', endpoint='#select_language'),
            Subdomain('<string(length=2):lang_code>', [
                Rule('/', endpoint='index'),
                Rule('/about', endpoint='about'),
                Rule('/help', endpoint='help')
            ])
        ])

    All the rules except for the ``'#select_language'`` endpoint will now
    listen on a two letter long subdomain that holds the language code
    for the current request.
    """

    def __init__(self, subdomain: str, rules: t.Iterable[RuleFactory]) -> None:
        self.subdomain = subdomain
        self.rules = rules

    def get_rules(self, map: "Map") -> t.Iterator["Rule"]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.subdomain = self.subdomain
                yield rule


class Submount(RuleFactory):
    """Like `Subdomain` but prefixes the URL rule with a given string::

        url_map = Map([
            Rule('/', endpoint='index'),
            Submount('/blog', [
                Rule('/', endpoint='blog/index'),
                Rule('/entry/<entry_slug>', endpoint='blog/show')
            ])
        ])

    Now the rule ``'blog/show'`` matches ``/blog/entry/<entry_slug>``.
    """

    def __init__(self, path: str, rules: t.Iterable[RuleFactory]) -> None:
        self.path = path.rstrip("/")
        self.rules = rules

    def get_rules(self, map: "Map") -> t.Iterator["Rule"]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.rule = self.path + rule.rule
                yield rule


class EndpointPrefix(RuleFactory):
    """Prefixes all endpoints (which must be strings for this factory) with
    another string. This can be useful for sub applications::

        url_map = Map([
            Rule('/', endpoint='index'),
            EndpointPrefix('blog/', [Submount('/blog', [
                Rule('/', endpoint='index'),
                Rule('/entry/<entry_slug>', endpoint='show')
            ])])
        ])
    """

    def __init__(self, prefix: str, rules: t.Iterable[RuleFactory]) -> None:
        self.prefix = prefix
        self.rules = rules

    def get_rules(self, map: "Map") -> t.Iterator["Rule"]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                rule = rule.empty()
                rule.endpoint = self.prefix + rule.endpoint
                yield rule


class RuleTemplate:
    """Returns copies of the rules wrapped and expands string templates in
    the endpoint, rule, defaults or subdomain sections.

    Here a small example for such a rule template::

        from werkzeug.routing import Map, Rule, RuleTemplate

        resource = RuleTemplate([
            Rule('/$name/', endpoint='$name.list'),
            Rule('/$name/<int:id>', endpoint='$name.show')
        ])

        url_map = Map([resource(name='user'), resource(name='page')])

    When a rule template is called the keyword arguments are used to
    replace the placeholders in all the string parameters.
    """

    def __init__(self, rules: t.Iterable["Rule"]) -> None:
        self.rules = list(rules)

    def __call__(self, *args: t.Any, **kwargs: t.Any) -> "RuleTemplateFactory":
        return RuleTemplateFactory(self.rules, dict(*args, **kwargs))


class RuleTemplateFactory(RuleFactory):
    """A factory that fills in template variables into rules.  Used by
    `RuleTemplate` internally.

    :internal:
    """

    def __init__(
        self, rules: t.Iterable[RuleFactory], context: t.Dict[str, t.Any]
    ) -> None:
        self.rules = rules
        self.context = context

    def get_rules(self, map: "Map") -> t.Iterator["Rule"]:
        for rulefactory in self.rules:
            for rule in rulefactory.get_rules(map):
                new_defaults = subdomain = None
                if rule.defaults:
                    new_defaults = {}
                    for key, value in rule.defaults.items():
                        if isinstance(value, str):
                            value = Template(value).substitute(self.context)
                        new_defaults[key] = value
                if rule.subdomain is not None:
                    subdomain = Template(rule.subdomain).substitute(self.context)
                new_endpoint = rule.endpoint
                if isinstance(new_endpoint, str):
                    new_endpoint = Template(new_endpoint).substitute(self.context)
                yield Rule(
                    Template(rule.rule).substitute(self.context),
                    new_defaults,
                    subdomain,
                    rule.methods,
                    rule.build_only,
                    new_endpoint,
                    rule.strict_slashes,
                )


def _prefix_names(src: str) -> ast.stmt:
    """ast parse and prefix names with `.` to avoid collision with user vars"""
    tree = ast.parse(src).body[0]
    if isinstance(tree, ast.Expr):
        tree = tree.value  # type: ignore
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = f".{node.id}"
    return tree


_CALL_CONVERTER_CODE_FMT = "self._converters[{elem!r}].to_url()"
_IF_KWARGS_URL_ENCODE_CODE = """\
if kwargs:
    q = '?'
    params = self._encode_query_vars(kwargs)
else:
    q = params = ''
"""
_IF_KWARGS_URL_ENCODE_AST = _prefix_names(_IF_KWARGS_URL_ENCODE_CODE)
_URL_ENCODE_AST_NAMES = (_prefix_names("q"), _prefix_names("params"))


class Rule(RuleFactory):
    """A Rule represents one URL pattern.  There are some options for `Rule`
    that change the way it behaves and are passed to the `Rule` constructor.
    Note that besides the rule-string all arguments *must* be keyword arguments
    in order to not break the application on Werkzeug upgrades.

    `string`
        Rule strings basically are just normal URL paths with placeholders in
        the format ``<converter(arguments):name>`` where the converter and the
        arguments are optional.  If no converter is defined the `default`
        converter is used which means `string` in the normal configuration.

        URL rules that end with a slash are branch URLs, others are leaves.
        If you have `strict_slashes` enabled (which is the default), all
        branch URLs that are matched without a trailing slash will trigger a
        redirect to the same URL with the missing slash appended.

        The converters are defined on the `Map`.

    `endpoint`
        The endpoint for this rule. This can be anything. A reference to a
        function, a string, a number etc.  The preferred way is using a string
        because the endpoint is used for URL generation.

    `defaults`
        An optional dict with defaults for other rules with the same endpoint.
        This is a bit tricky but useful if you want to have unique URLs::

            url_map = Map([
                Rule('/all/', defaults={'page': 1}, endpoint='all_entries'),
                Rule('/all/page/<int:page>', endpoint='all_entries')
            ])

        If a user now visits ``http://example.com/all/page/1`` he will be
        redirected to ``http://example.com/all/``.  If `redirect_defaults` is
        disabled on the `Map` instance this will only affect the URL
        generation.

    `subdomain`
        The subdomain rule string for this rule. If not specified the rule
        only matches for the `default_subdomain` of the map.  If the map is
        not bound to a subdomain this feature is disabled.

        Can be useful if you want to have user profiles on different subdomains
        and all subdomains are forwarded to your application::

            url_map = Map([
                Rule('/', subdomain='<username>', endpoint='user/homepage'),
                Rule('/stats', subdomain='<username>', endpoint='user/stats')
            ])

    `methods`
        A sequence of http methods this rule applies to.  If not specified, all
        methods are allowed. For example this can be useful if you want different
        endpoints for `POST` and `GET`.  If methods are defined and the path
        matches but the method matched against is not in this list or in the
        list of another rule for that path the error raised is of the type
        `MethodNotAllowed` rather than `NotFound`.  If `GET` is present in the
        list of methods and `HEAD` is not, `HEAD` is added automatically.

    `strict_slashes`
        Override the `Map` setting for `strict_slashes` only for this rule. If
        not specified the `Map` setting is used.

    `merge_slashes`
        Override :attr:`Map.merge_slashes` for this rule.

    `build_only`
        Set this to True and the rule will never match but will create a URL
        that can be build. This is useful if you have resources on a subdomain
        or folder that are not handled by the WSGI application (like static data)

    `redirect_to`
        If given this must be either a string or callable.  In case of a
        callable it's called with the url adapter that triggered the match and
        the values of the URL as keyword arguments and has to return the target
        for the redirect, otherwise it has to be a string with placeholders in
        rule syntax::

            def foo_with_slug(adapter, id):
                # ask the database for the slug for the old id.  this of
                # course has nothing to do with werkzeug.
                return f'foo/{Foo.get_slug_for_id(id)}'

            url_map = Map([
                Rule('/foo/<slug>', endpoint='foo'),
                Rule('/some/old/url/<slug>', redirect_to='foo/<slug>'),
                Rule('/other/old/url/<int:id>', redirect_to=foo_with_slug)
            ])

        When the rule is matched the routing system will raise a
        `RequestRedirect` exception with the target for the redirect.

        Keep in mind that the URL will be joined against the URL root of the
        script so don't use a leading slash on the target URL unless you
        really mean root of that domain.

    `alias`
        If enabled this rule serves as an alias for another rule with the same
        endpoint and arguments.

    `host`
        If provided and the URL map has host matching enabled this can be
        used to provide a match rule for the whole host.  This also means
        that the subdomain feature is disabled.

    `websocket`
        If ``True``, this rule is only matches for WebSocket (``ws://``,
        ``wss://``) requests. By default, rules will only match for HTTP
        requests.

    .. versionadded:: 1.0
        Added ``websocket``.

    .. versionadded:: 1.0
        Added ``merge_slashes``.

    .. versionadded:: 0.7
        Added ``alias`` and ``host``.

    .. versionchanged:: 0.6.1
       ``HEAD`` is added to ``methods`` if ``GET`` is present.
    """

    def __init__(
        self,
        string: str,
        defaults: t.Optional[t.Mapping[str, t.Any]] = None,
        subdomain: t.Optional[str] = None,
        methods: t.Optional[t.Iterable[str]] = None,
        build_only: bool = False,
        endpoint: t.Optional[str] = None,
        strict_slashes: t.Optional[bool] = None,
        merge_slashes: t.Optional[bool] = None,
        redirect_to: t.Optional[t.Union[str, t.Callable[..., str]]] = None,
        alias: bool = False,
        host: t.Optional[str] = None,
        websocket: bool = False,
    ) -> None:
        if not string.startswith("/"):
            raise ValueError("urls must start with a leading slash")
        self.rule = string
        self.is_leaf = not string.endswith("/")

        self.map: "Map" = None  # type: ignore
        self.strict_slashes = strict_slashes
        self.merge_slashes = merge_slashes
        self.subdomain = subdomain
        self.host = host
        self.defaults = defaults
        self.build_only = build_only
        self.alias = alias
        self.websocket = websocket

        if methods is not None:
            if isinstance(methods, str):
                raise TypeError("'methods' should be a list of strings.")

            methods = {x.upper() for x in methods}

            if "HEAD" not in methods and "GET" in methods:
                methods.add("HEAD")

            if websocket and methods - {"GET", "HEAD", "OPTIONS"}:
                raise ValueError(
                    "WebSocket rules can only use 'GET', 'HEAD', and 'OPTIONS' methods."
                )

        self.methods = methods
        self.endpoint: str = endpoint  # type: ignore
        self.redirect_to = redirect_to

        if defaults:
            self.arguments = set(map(str, defaults))
        else:
            self.arguments = set()

        self._trace: t.List[t.Tuple[bool, str]] = []

    def empty(self) -> "Rule":
        """
        Return an unbound copy of this rule.

        This can be useful if want to reuse an already bound URL for another
        map.  See ``get_empty_kwargs`` to override what keyword arguments are
        provided to the new copy.
        """
        return type(self)(self.rule, **self.get_empty_kwargs())

    def get_empty_kwargs(self) -> t.Mapping[str, t.Any]:
        """
        Provides kwargs for instantiating empty copy with empty()

        Use this method to provide custom keyword arguments to the subclass of
        ``Rule`` when calling ``some_rule.empty()``.  Helpful when the subclass
        has custom keyword arguments that are needed at instantiation.

        Must return a ``dict`` that will be provided as kwargs to the new
        instance of ``Rule``, following the initial ``self.rule`` value which
        is always provided as the first, required positional argument.
        """
        defaults = None
        if self.defaults:
            defaults = dict(self.defaults)
        return dict(
            defaults=defaults,
            subdomain=self.subdomain,
            methods=self.methods,
            build_only=self.build_only,
            endpoint=self.endpoint,
            strict_slashes=self.strict_slashes,
            redirect_to=self.redirect_to,
            alias=self.alias,
            host=self.host,
        )

    def get_rules(self, map: "Map") -> t.Iterator["Rule"]:
        yield self

    def refresh(self) -> None:
        """Rebinds and refreshes the URL.  Call this if you modified the
        rule in place.

        :internal:
        """
        self.bind(self.map, rebind=True)

    def bind(self, map: "Map", rebind: bool = False) -> None:
        """Bind the url to a map and create a regular expression based on
        the information from the rule itself and the defaults from the map.

        :internal:
        """
        if self.map is not None and not rebind:
            raise RuntimeError(f"url rule {self!r} already bound to map {self.map!r}")
        self.map = map
        if self.strict_slashes is None:
            self.strict_slashes = map.strict_slashes
        if self.merge_slashes is None:
            self.merge_slashes = map.merge_slashes
        if self.subdomain is None:
            self.subdomain = map.default_subdomain
        self.compile()

    def get_converter(
        self,
        variable_name: str,
        converter_name: str,
        args: t.Tuple,
        kwargs: t.Mapping[str, t.Any],
    ) -> "BaseConverter":
        """Looks up the converter for the given parameter.

        .. versionadded:: 0.9
        """
        if converter_name not in self.map.converters:
            raise LookupError(f"the converter {converter_name!r} does not exist")
        return self.map.converters[converter_name](self.map, *args, **kwargs)

    def _encode_query_vars(self, query_vars: t.Mapping[str, t.Any]) -> str:
        return url_encode(
            query_vars,
            charset=self.map.charset,
            sort=self.map.sort_parameters,
            key=self.map.sort_key,
        )

    def compile(self) -> None:
        """Compiles the regular expression and stores it."""
        assert self.map is not None, "rule not bound"

        if self.map.host_matching:
            domain_rule = self.host or ""
        else:
            domain_rule = self.subdomain or ""

        self._trace = []
        self._converters: t.Dict[str, "BaseConverter"] = {}
        self._static_weights: t.List[t.Tuple[int, int]] = []
        self._argument_weights: t.List[int] = []
        regex_parts = []

        def _build_regex(rule: str) -> None:
            index = 0
            for converter, arguments, variable in parse_rule(rule):
                if converter is None:
                    for match in re.finditer(r"/+|[^/]+", variable):
                        part = match.group(0)
                        if part.startswith("/"):
                            if self.merge_slashes:
                                regex_parts.append(r"/+?")
                                self._trace.append((False, "/"))
                            else:
                                regex_parts.append(part)
                                self._trace.append((False, part))
                            continue
                        self._trace.append((False, part))
                        regex_parts.append(re.escape(part))
                        if part:
                            self._static_weights.append((index, -len(part)))
                else:
                    if arguments:
                        c_args, c_kwargs = parse_converter_args(arguments)
                    else:
                        c_args = ()
                        c_kwargs = {}
                    convobj = self.get_converter(variable, converter, c_args, c_kwargs)
                    regex_parts.append(f"(?P<{variable}>{convobj.regex})")
                    self._converters[variable] = convobj
                    self._trace.append((True, variable))
                    self._argument_weights.append(convobj.weight)
                    self.arguments.add(str(variable))
                index = index + 1

        _build_regex(domain_rule)
        regex_parts.append("\\|")
        self._trace.append((False, "|"))
        _build_regex(self.rule if self.is_leaf else self.rule.rstrip("/"))
        if not self.is_leaf:
            self._trace.append((False, "/"))

        self._build: t.Callable[..., t.Tuple[str, str]]
        self._build = self._compile_builder(False).__get__(self, None)  # type: ignore
        self._build_unknown: t.Callable[..., t.Tuple[str, str]]
        self._build_unknown = self._compile_builder(True).__get__(  # type: ignore
            self, None
        )

        if self.build_only:
            return

        if not (self.is_leaf and self.strict_slashes):
            reps = "*" if self.merge_slashes else "?"
            tail = f"(?<!/)(?P<__suffix__>/{reps})"
        else:
            tail = ""

        regex = f"^{''.join(regex_parts)}{tail}$"
        self._regex = re.compile(regex)

    def match(
        self, path: str, method: t.Optional[str] = None
    ) -> t.Optional[t.MutableMapping[str, t.Any]]:
        """Check if the rule matches a given path. Path is a string in the
        form ``"subdomain|/path"`` and is assembled by the map.  If
        the map is doing host matching the subdomain part will be the host
        instead.

        If the rule matches a dict with the converted values is returned,
        otherwise the return value is `None`.

        :internal:
        """
        if not self.build_only:
            require_redirect = False

            m = self._regex.search(path)
            if m is not None:
                groups = m.groupdict()
                # we have a folder like part of the url without a trailing
                # slash and strict slashes enabled. raise an exception that
                # tells the map to redirect to the same url but with a
                # trailing slash
                if (
                    self.strict_slashes
                    and not self.is_leaf
                    and not groups.pop("__suffix__")
                    and (
                        method is None or self.methods is None or method in self.methods
                    )
                ):
                    path += "/"
                    require_redirect = True
                # if we are not in strict slashes mode we have to remove
                # a __suffix__
                elif not self.strict_slashes:
                    del groups["__suffix__"]

                result = {}
                for name, value in groups.items():
                    try:
                        value = self._converters[name].to_python(value)
                    except ValidationError:
                        return None
                    result[str(name)] = value
                if self.defaults:
                    result.update(self.defaults)

                if self.merge_slashes:
                    new_path = "|".join(self.build(result, False))  # type: ignore
                    if path.endswith("/") and not new_path.endswith("/"):
                        new_path += "/"
                    if new_path.count("/") < path.count("/"):
                        # The URL will be encoded when MapAdapter.match
                        # handles the RequestPath raised below. Decode
                        # the URL here to avoid a double encoding.
                        path = url_unquote(new_path)
                        require_redirect = True

                if require_redirect:
                    path = path.split("|", 1)[1]
                    raise RequestPath(path)

                if self.alias and self.map.redirect_defaults:
                    raise RequestAliasRedirect(result)

                return result

        return None

    @staticmethod
    def _get_func_code(code: CodeType, name: str) -> t.Callable[..., t.Tuple[str, str]]:
        globs: t.Dict[str, t.Any] = {}
        locs: t.Dict[str, t.Any] = {}
        exec(code, globs, locs)
        return locs[name]  # type: ignore

    def _compile_builder(
        self, append_unknown: bool = True
    ) -> t.Callable[..., t.Tuple[str, str]]:
        defaults = self.defaults or {}
        dom_ops: t.List[t.Tuple[bool, str]] = []
        url_ops: t.List[t.Tuple[bool, str]] = []

        opl = dom_ops
        for is_dynamic, data in self._trace:
            if data == "|" and opl is dom_ops:
                opl = url_ops
                continue
            # this seems like a silly case to ever come up but:
            # if a default is given for a value that appears in the rule,
            # resolve it to a constant ahead of time
            if is_dynamic and data in defaults:
                data = self._converters[data].to_url(defaults[data])
                opl.append((False, data))
            elif not is_dynamic:
                opl.append(
                    (False, url_quote(_to_bytes(data, self.map.charset), safe="/:|+"))
                )
            else:
                opl.append((True, data))

        def _convert(elem: str) -> ast.stmt:
            ret = _prefix_names(_CALL_CONVERTER_CODE_FMT.format(elem=elem))
            ret.args = [ast.Name(str(elem), ast.Load())]  # type: ignore  # str for py2
            return ret

        def _parts(ops: t.List[t.Tuple[bool, str]]) -> t.List[ast.AST]:
            parts = [
                _convert(elem) if is_dynamic else ast.Str(s=elem)
                for is_dynamic, elem in ops
            ]
            parts = parts or [ast.Str("")]
            # constant fold
            ret = [parts[0]]
            for p in parts[1:]:
                if isinstance(p, ast.Str) and isinstance(ret[-1], ast.Str):
                    ret[-1] = ast.Str(ret[-1].s + p.s)
                else:
                    ret.append(p)
            return ret

        dom_parts = _parts(dom_ops)
        url_parts = _parts(url_ops)
        if not append_unknown:
            body = []
        else:
            body = [_IF_KWARGS_URL_ENCODE_AST]
            url_parts.extend(_URL_ENCODE_AST_NAMES)

        def _join(parts: t.List[ast.AST]) -> ast.AST:
            if len(parts) == 1:  # shortcut
                return parts[0]
            return ast.JoinedStr(parts)

        body.append(
            ast.Return(ast.Tuple([_join(dom_parts), _join(url_parts)], ast.Load()))
        )

        pargs = [
            elem
            for is_dynamic, elem in dom_ops + url_ops
            if is_dynamic and elem not in defaults
        ]
        kargs = [str(k) for k in defaults]

        func_ast: ast.FunctionDef = _prefix_names("def _(): pass")  # type: ignore
        func_ast.name = f"<builder:{self.rule!r}>"
        func_ast.args.args.append(ast.arg(".self", None))
        for arg in pargs + kargs:
            func_ast.args.args.append(ast.arg(arg, None))
        func_ast.args.kwarg = ast.arg(".kwargs", None)
        for _ in kargs:
            func_ast.args.defaults.append(ast.Str(""))
        func_ast.body = body

        # use `ast.parse` instead of `ast.Module` for better portability
        # Python 3.8 changes the signature of `ast.Module`
        module = ast.parse("")
        module.body = [func_ast]

        # mark everything as on line 1, offset 0
        # less error-prone than `ast.fix_missing_locations`
        # bad line numbers cause an assert to fail in debug builds
        for node in ast.walk(module):
            if "lineno" in node._attributes:
                node.lineno = 1
            if "col_offset" in node._attributes:
                node.col_offset = 0

        code = compile(module, "<werkzeug routing>", "exec")
        return self._get_func_code(code, func_ast.name)

    def build(
        self, values: t.Mapping[str, t.Any], append_unknown: bool = True
    ) -> t.Optional[t.Tuple[str, str]]:
        """Assembles the relative url for that rule and the subdomain.
        If building doesn't work for some reasons `None` is returned.

        :internal:
        """
        try:
            if append_unknown:
                return self._build_unknown(**values)
            else:
                return self._build(**values)
        except ValidationError:
            return None

    def provides_defaults_for(self, rule: "Rule") -> bool:
        """Check if this rule has defaults for a given rule.

        :internal:
        """
        return bool(
            not self.build_only
            and self.defaults
            and self.endpoint == rule.endpoint
            and self != rule
            and self.arguments == rule.arguments
        )

    def suitable_for(
        self, values: t.Mapping[str, t.Any], method: t.Optional[str] = None
    ) -> bool:
        """Check if the dict of values has enough data for url generation.

        :internal:
        """
        # if a method was given explicitly and that method is not supported
        # by this rule, this rule is not suitable.
        if (
            method is not None
            and self.methods is not None
            and method not in self.methods
        ):
            return False

        defaults = self.defaults or ()

        # all arguments required must be either in the defaults dict or
        # the value dictionary otherwise it's not suitable
        for key in self.arguments:
            if key not in defaults and key not in values:
                return False

        # in case defaults are given we ensure that either the value was
        # skipped or the value is the same as the default value.
        if defaults:
            for key, value in defaults.items():
                if key in values and value != values[key]:
                    return False

        return True

    def match_compare_key(
        self,
    ) -> t.Tuple[bool, int, t.Iterable[t.Tuple[int, int]], int, t.Iterable[int]]:
        """The match compare key for sorting.

        Current implementation:

        1.  rules without any arguments come first for performance
            reasons only as we expect them to match faster and some
            common ones usually don't have any arguments (index pages etc.)
        2.  rules with more static parts come first so the second argument
            is the negative length of the number of the static weights.
        3.  we order by static weights, which is a combination of index
            and length
        4.  The more complex rules come first so the next argument is the
            negative length of the number of argument weights.
        5.  lastly we order by the actual argument weights.

        :internal:
        """
        return (
            bool(self.arguments),
            -len(self._static_weights),
            self._static_weights,
            -len(self._argument_weights),
            self._argument_weights,
        )

    def build_compare_key(self) -> t.Tuple[int, int, int]:
        """The build compare key for sorting.

        :internal:
        """
        return (1 if self.alias else 0, -len(self.arguments), -len(self.defaults or ()))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._trace == other._trace

    __hash__ = None  # type: ignore

    def __str__(self) -> str:
        return self.rule

    def __repr__(self) -> str:
        if self.map is None:
            return f"<{type(self).__name__} (unbound)>"
        parts = []
        for is_dynamic, data in self._trace:
            if is_dynamic:
                parts.append(f"<{data}>")
            else:
                parts.append(data)
        parts = "".join(parts).lstrip("|")
        methods = f" ({', '.join(self.methods)})" if self.methods is not None else ""
        return f"<{type(self).__name__} {parts!r}{methods} -> {self.endpoint}>"


class BaseConverter:
    """Base class for all converters."""

    regex = "[^/]+"
    weight = 100

    def __init__(self, map: "Map", *args: t.Any, **kwargs: t.Any) -> None:
        self.map = map

    def to_python(self, value: str) -> t.Any:
        return value

    def to_url(self, value: t.Any) -> str:
        if isinstance(value, (bytes, bytearray)):
            return _fast_url_quote(value)
        return _fast_url_quote(str(value).encode(self.map.charset))


class UnicodeConverter(BaseConverter):
    """This converter is the default converter and accepts any string but
    only one path segment.  Thus the string can not include a slash.

    This is the default validator.

    Example::

        Rule('/pages/<page>'),
        Rule('/<string(length=2):lang_code>')

    :param map: the :class:`Map`.
    :param minlength: the minimum length of the string.  Must be greater
                      or equal 1.
    :param maxlength: the maximum length of the string.
    :param length: the exact length of the string.
    """

    def __init__(
        self,
        map: "Map",
        minlength: int = 1,
        maxlength: t.Optional[int] = None,
        length: t.Optional[int] = None,
    ) -> None:
        super().__init__(map)
        if length is not None:
            length_regex = f"{{{int(length)}}}"
        else:
            if maxlength is None:
                maxlength_value = ""
            else:
                maxlength_value = str(int(maxlength))
            length_regex = f"{{{int(minlength)},{maxlength_value}}}"
        self.regex = f"[^/]{length_regex}"


class AnyConverter(BaseConverter):
    """Matches one of the items provided.  Items can either be Python
    identifiers or strings::

        Rule('/<any(about, help, imprint, class, "foo,bar"):page_name>')

    :param map: the :class:`Map`.
    :param items: this function accepts the possible items as positional
                  arguments.
    """

    def __init__(self, map: "Map", *items: str) -> None:
        super().__init__(map)
        self.regex = f"(?:{'|'.join([re.escape(x) for x in items])})"


class PathConverter(BaseConverter):
    """Like the default :class:`UnicodeConverter`, but it also matches
    slashes.  This is useful for wikis and similar applications::

        Rule('/<path:wikipage>')
        Rule('/<path:wikipage>/edit')

    :param map: the :class:`Map`.
    """

    regex = "[^/].*?"
    weight = 200


class NumberConverter(BaseConverter):
    """Baseclass for `IntegerConverter` and `FloatConverter`.

    :internal:
    """

    weight = 50
    num_convert: t.Callable = int

    def __init__(
        self,
        map: "Map",
        fixed_digits: int = 0,
        min: t.Optional[int] = None,
        max: t.Optional[int] = None,
        signed: bool = False,
    ) -> None:
        if signed:
            self.regex = self.signed_regex
        super().__init__(map)
        self.fixed_digits = fixed_digits
        self.min = min
        self.max = max
        self.signed = signed

    def to_python(self, value: str) -> t.Any:
        if self.fixed_digits and len(value) != self.fixed_digits:
            raise ValidationError()
        value = self.num_convert(value)
        if (self.min is not None and value < self.min) or (
            self.max is not None and value > self.max
        ):
            raise ValidationError()
        return value

    def to_url(self, value: t.Any) -> str:
        value = str(self.num_convert(value))
        if self.fixed_digits:
            value = value.zfill(self.fixed_digits)
        return value

    @property
    def signed_regex(self) -> str:
        return f"-?{self.regex}"


class IntegerConverter(NumberConverter):
    """This converter only accepts integer values::

        Rule("/page/<int:page>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/page/<int(signed=True):page>")

    :param map: The :class:`Map`.
    :param fixed_digits: The number of fixed digits in the URL. If you
        set this to ``4`` for example, the rule will only match if the
        URL looks like ``/0001/``. The default is variable length.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """

    regex = r"\d+"


class FloatConverter(NumberConverter):
    """This converter only accepts floating point values::

        Rule("/probability/<float:probability>")

    By default it only accepts unsigned, positive values. The ``signed``
    parameter will enable signed, negative values. ::

        Rule("/offset/<float(signed=True):offset>")

    :param map: The :class:`Map`.
    :param min: The minimal value.
    :param max: The maximal value.
    :param signed: Allow signed (negative) values.

    .. versionadded:: 0.15
        The ``signed`` parameter.
    """

    regex = r"\d+\.\d+"
    num_convert = float

    def __init__(
        self,
        map: "Map",
        min: t.Optional[float] = None,
        max: t.Optional[float] = None,
        signed: bool = False,
    ) -> None:
        super().__init__(map, min=min, max=max, signed=signed)  # type: ignore


class UUIDConverter(BaseConverter):
    """This converter only accepts UUID strings::

        Rule('/object/<uuid:identifier>')

    .. versionadded:: 0.10

    :param map: the :class:`Map`.
    """

    regex = (
        r"[A-Fa-f0-9]{8}-[A-Fa-f0-9]{4}-"
        r"[A-Fa-f0-9]{4}-[A-Fa-f0-9]{4}-[A-Fa-f0-9]{12}"
    )

    def to_python(self, value: str) -> uuid.UUID:
        return uuid.UUID(value)

    def to_url(self, value: uuid.UUID) -> str:
        return str(value)


#: the default converter mapping for the map.
DEFAULT_CONVERTERS: t.Mapping[str, t.Type[BaseConverter]] = {
    "default": UnicodeConverter,
    "string": UnicodeConverter,
    "any": AnyConverter,
    "path": PathConverter,
    "int": IntegerConverter,
    "float": FloatConverter,
    "uuid": UUIDConverter,
}


class Map:
    """The map class stores all the URL rules and some configuration
    parameters.  Some of the configuration values are only stored on the
    `Map` instance since those affect all rules, others are just defaults
    and can be overridden for each rule.  Note that you have to specify all
    arguments besides the `rules` as keyword arguments!

    :param rules: sequence of url rules for this map.
    :param default_subdomain: The default subdomain for rules without a
                              subdomain defined.
    :param charset: charset of the url. defaults to ``"utf-8"``
    :param strict_slashes: If a rule ends with a slash but the matched
        URL does not, redirect to the URL with a trailing slash.
    :param merge_slashes: Merge consecutive slashes when matching or
        building URLs. Matches will redirect to the normalized URL.
        Slashes in variable parts are not merged.
    :param redirect_defaults: This will redirect to the default rule if it
                              wasn't visited that way. This helps creating
                              unique URLs.
    :param converters: A dict of converters that adds additional converters
                       to the list of converters. If you redefine one
                       converter this will override the original one.
    :param sort_parameters: If set to `True` the url parameters are sorted.
                            See `url_encode` for more details.
    :param sort_key: The sort key function for `url_encode`.
    :param encoding_errors: the error method to use for decoding
    :param host_matching: if set to `True` it enables the host matching
                          feature and disables the subdomain one.  If
                          enabled the `host` parameter to rules is used
                          instead of the `subdomain` one.

    .. versionchanged:: 1.0
        If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules
        will match.

    .. versionchanged:: 1.0
        Added ``merge_slashes``.

    .. versionchanged:: 0.7
        Added ``encoding_errors`` and ``host_matching``.

    .. versionchanged:: 0.5
        Added ``sort_parameters`` and ``sort_key``.
    """

    #: A dict of default converters to be used.
    default_converters = ImmutableDict(DEFAULT_CONVERTERS)

    #: The type of lock to use when updating.
    #:
    #: .. versionadded:: 1.0
    lock_class = Lock

    def __init__(
        self,
        rules: t.Optional[t.Iterable[RuleFactory]] = None,
        default_subdomain: str = "",
        charset: str = "utf-8",
        strict_slashes: bool = True,
        merge_slashes: bool = True,
        redirect_defaults: bool = True,
        converters: t.Optional[t.Mapping[str, t.Type[BaseConverter]]] = None,
        sort_parameters: bool = False,
        sort_key: t.Optional[t.Callable[[t.Any], t.Any]] = None,
        encoding_errors: str = "replace",
        host_matching: bool = False,
    ) -> None:
        self._rules: t.List[Rule] = []
        self._rules_by_endpoint: t.Dict[str, t.List[Rule]] = {}
        self._remap = True
        self._remap_lock = self.lock_class()

        self.default_subdomain = default_subdomain
        self.charset = charset
        self.encoding_errors = encoding_errors
        self.strict_slashes = strict_slashes
        self.merge_slashes = merge_slashes
        self.redirect_defaults = redirect_defaults
        self.host_matching = host_matching

        self.converters = self.default_converters.copy()
        if converters:
            self.converters.update(converters)

        self.sort_parameters = sort_parameters
        self.sort_key = sort_key

        for rulefactory in rules or ():
            self.add(rulefactory)

    def is_endpoint_expecting(self, endpoint: str, *arguments: str) -> bool:
        """Iterate over all rules and check if the endpoint expects
        the arguments provided.  This is for example useful if you have
        some URLs that expect a language code and others that do not and
        you want to wrap the builder a bit so that the current language
        code is automatically added if not provided but endpoints expect
        it.

        :param endpoint: the endpoint to check.
        :param arguments: this function accepts one or more arguments
                          as positional arguments.  Each one of them is
                          checked.
        """
        self.update()
        arguments = set(arguments)
        for rule in self._rules_by_endpoint[endpoint]:
            if arguments.issubset(rule.arguments):
                return True
        return False

    def iter_rules(self, endpoint: t.Optional[str] = None) -> t.Iterator[Rule]:
        """Iterate over all rules or the rules of an endpoint.

        :param endpoint: if provided only the rules for that endpoint
                         are returned.
        :return: an iterator
        """
        self.update()
        if endpoint is not None:
            return iter(self._rules_by_endpoint[endpoint])
        return iter(self._rules)

    def add(self, rulefactory: RuleFactory) -> None:
        """Add a new rule or factory to the map and bind it.  Requires that the
        rule is not bound to another map.

        :param rulefactory: a :class:`Rule` or :class:`RuleFactory`
        """
        for rule in rulefactory.get_rules(self):
            rule.bind(self)
            self._rules.append(rule)
            self._rules_by_endpoint.setdefault(rule.endpoint, []).append(rule)
        self._remap = True

    def bind(
        self,
        server_name: str,
        script_name: t.Optional[str] = None,
        subdomain: t.Optional[str] = None,
        url_scheme: str = "http",
        default_method: str = "GET",
        path_info: t.Optional[str] = None,
        query_args: t.Optional[t.Union[t.Mapping[str, t.Any], str]] = None,
    ) -> "MapAdapter":
        """Return a new :class:`MapAdapter` with the details specified to the
        call.  Note that `script_name` will default to ``'/'`` if not further
        specified or `None`.  The `server_name` at least is a requirement
        because the HTTP RFC requires absolute URLs for redirects and so all
        redirect exceptions raised by Werkzeug will contain the full canonical
        URL.

        If no path_info is passed to :meth:`match` it will use the default path
        info passed to bind.  While this doesn't really make sense for
        manual bind calls, it's useful if you bind a map to a WSGI
        environment which already contains the path info.

        `subdomain` will default to the `default_subdomain` for this map if
        no defined. If there is no `default_subdomain` you cannot use the
        subdomain feature.

        .. versionchanged:: 1.0
            If ``url_scheme`` is ``ws`` or ``wss``, only WebSocket rules
            will match.

        .. versionchanged:: 0.15
            ``path_info`` defaults to ``'/'`` if ``None``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionchanged:: 0.7
            Added ``query_args``.
        """
        server_name = server_name.lower()
        if self.host_matching:
            if subdomain is not None:
                raise RuntimeError("host matching enabled and a subdomain was provided")
        elif subdomain is None:
            subdomain = self.default_subdomain
        if script_name is None:
            script_name = "/"
        if path_info is None:
            path_info = "/"

        try:
            server_name = _encode_idna(server_name)  # type: ignore
        except UnicodeError as e:
            raise BadHost() from e

        return MapAdapter(
            self,
            server_name,
            script_name,
            subdomain,
            url_scheme,
            path_info,
            default_method,
            query_args,
        )

    def bind_to_environ(
        self,
        environ: "WSGIEnvironment",
        server_name: t.Optional[str] = None,
        subdomain: t.Optional[str] = None,
    ) -> "MapAdapter":
        """Like :meth:`bind` but you can pass it an WSGI environment and it
        will fetch the information from that dictionary.  Note that because of
        limitations in the protocol there is no way to get the current
        subdomain and real `server_name` from the environment.  If you don't
        provide it, Werkzeug will use `SERVER_NAME` and `SERVER_PORT` (or
        `HTTP_HOST` if provided) as used `server_name` with disabled subdomain
        feature.

        If `subdomain` is `None` but an environment and a server name is
        provided it will calculate the current subdomain automatically.
        Example: `server_name` is ``'example.com'`` and the `SERVER_NAME`
        in the wsgi `environ` is ``'staging.dev.example.com'`` the calculated
        subdomain will be ``'staging.dev'``.

        If the object passed as environ has an environ attribute, the value of
        this attribute is used instead.  This allows you to pass request
        objects.  Additionally `PATH_INFO` added as a default of the
        :class:`MapAdapter` so that you don't have to pass the path info to
        the match method.

        .. versionchanged:: 1.0.0
            If the passed server name specifies port 443, it will match
            if the incoming scheme is ``https`` without a port.

        .. versionchanged:: 1.0.0
            A warning is shown when the passed server name does not
            match the incoming WSGI server name.

        .. versionchanged:: 0.8
           This will no longer raise a ValueError when an unexpected server
           name was passed.

        .. versionchanged:: 0.5
            previously this method accepted a bogus `calculate_subdomain`
            parameter that did not have any effect.  It was removed because
            of that.

        :param environ: a WSGI environment.
        :param server_name: an optional server name hint (see above).
        :param subdomain: optionally the current subdomain (see above).
        """
        environ = _get_environ(environ)
        wsgi_server_name = get_host(environ).lower()
        scheme = environ["wsgi.url_scheme"]
        upgrade = any(
            v.strip() == "upgrade"
            for v in environ.get("HTTP_CONNECTION", "").lower().split(",")
        )

        if upgrade and environ.get("HTTP_UPGRADE", "").lower() == "websocket":
            scheme = "wss" if scheme == "https" else "ws"

        if server_name is None:
            server_name = wsgi_server_name
        else:
            server_name = server_name.lower()

            # strip standard port to match get_host()
            if scheme in {"http", "ws"} and server_name.endswith(":80"):
                server_name = server_name[:-3]
            elif scheme in {"https", "wss"} and server_name.endswith(":443"):
                server_name = server_name[:-4]

        if subdomain is None and not self.host_matching:
            cur_server_name = wsgi_server_name.split(".")
            real_server_name = server_name.split(".")
            offset = -len(real_server_name)

            if cur_server_name[offset:] != real_server_name:
                # This can happen even with valid configs if the server was
                # accessed directly by IP address under some situations.
                # Instead of raising an exception like in Werkzeug 0.7 or
                # earlier we go by an invalid subdomain which will result
                # in a 404 error on matching.
                warnings.warn(
                    f"Current server name {wsgi_server_name!r} doesn't match configured"
                    f" server name {server_name!r}",
                    stacklevel=2,
                )
                subdomain = "<invalid>"
            else:
                subdomain = ".".join(filter(None, cur_server_name[:offset]))

        def _get_wsgi_string(name: str) -> t.Optional[str]:
            val = environ.get(name)
            if val is not None:
                return _wsgi_decoding_dance(val, self.charset)
            return None

        script_name = _get_wsgi_string("SCRIPT_NAME")
        path_info = _get_wsgi_string("PATH_INFO")
        query_args = _get_wsgi_string("QUERY_STRING")
        return Map.bind(
            self,
            server_name,
            script_name,
            subdomain,
            scheme,
            environ["REQUEST_METHOD"],
            path_info,
            query_args=query_args,
        )

    def update(self) -> None:
        """Called before matching and building to keep the compiled rules
        in the correct order after things changed.
        """
        if not self._remap:
            return

        with self._remap_lock:
            if not self._remap:
                return

            self._rules.sort(key=lambda x: x.match_compare_key())
            for rules in self._rules_by_endpoint.values():
                rules.sort(key=lambda x: x.build_compare_key())
            self._remap = False

    def __repr__(self) -> str:
        rules = self.iter_rules()
        return f"{type(self).__name__}({pformat(list(rules))})"


class MapAdapter:

    """Returned by :meth:`Map.bind` or :meth:`Map.bind_to_environ` and does
    the URL matching and building based on runtime information.
    """

    def __init__(
        self,
        map: Map,
        server_name: str,
        script_name: str,
        subdomain: t.Optional[str],
        url_scheme: str,
        path_info: str,
        default_method: str,
        query_args: t.Optional[t.Union[t.Mapping[str, t.Any], str]] = None,
    ):
        self.map = map
        self.server_name = _to_str(server_name)
        script_name = _to_str(script_name)
        if not script_name.endswith("/"):
            script_name += "/"
        self.script_name = script_name
        self.subdomain = _to_str(subdomain)
        self.url_scheme = _to_str(url_scheme)
        self.path_info = _to_str(path_info)
        self.default_method = _to_str(default_method)
        self.query_args = query_args
        self.websocket = self.url_scheme in {"ws", "wss"}

    def dispatch(
        self,
        view_func: t.Callable[[str, t.Mapping[str, t.Any]], "WSGIApplication"],
        path_info: t.Optional[str] = None,
        method: t.Optional[str] = None,
        catch_http_exceptions: bool = False,
    ) -> "WSGIApplication":
        """Does the complete dispatching process.  `view_func` is called with
        the endpoint and a dict with the values for the view.  It should
        look up the view function, call it, and return a response object
        or WSGI application.  http exceptions are not caught by default
        so that applications can display nicer error messages by just
        catching them by hand.  If you want to stick with the default
        error messages you can pass it ``catch_http_exceptions=True`` and
        it will catch the http exceptions.

        Here a small example for the dispatch usage::

            from werkzeug.wrappers import Request, Response
            from werkzeug.wsgi import responder
            from werkzeug.routing import Map, Rule

            def on_index(request):
                return Response('Hello from the index')

            url_map = Map([Rule('/', endpoint='index')])
            views = {'index': on_index}

            @responder
            def application(environ, start_response):
                request = Request(environ)
                urls = url_map.bind_to_environ(environ)
                return urls.dispatch(lambda e, v: views[e](request, **v),
                                     catch_http_exceptions=True)

        Keep in mind that this method might return exception objects, too, so
        use :class:`Response.force_type` to get a response object.

        :param view_func: a function that is called with the endpoint as
                          first argument and the value dict as second.  Has
                          to dispatch to the actual view function with this
                          information.  (see above)
        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param catch_http_exceptions: set to `True` to catch any of the
                                      werkzeug :class:`HTTPException`\\s.
        """
        try:
            try:
                endpoint, args = self.match(path_info, method)
            except RequestRedirect as e:
                return e
            return view_func(endpoint, args)
        except HTTPException as e:
            if catch_http_exceptions:
                return e
            raise

    @typing.overload
    def match(  # type: ignore
        self,
        path_info: t.Optional[str] = None,
        method: t.Optional[str] = None,
        return_rule: "te.Literal[False]" = False,
        query_args: t.Optional[t.Union[t.Mapping[str, t.Any], str]] = None,
        websocket: t.Optional[bool] = None,
    ) -> t.Tuple[str, t.Mapping[str, t.Any]]:
        ...

    @typing.overload
    def match(
        self,
        path_info: t.Optional[str] = None,
        method: t.Optional[str] = None,
        return_rule: "te.Literal[True]" = True,
        query_args: t.Optional[t.Union[t.Mapping[str, t.Any], str]] = None,
        websocket: t.Optional[bool] = None,
    ) -> t.Tuple[Rule, t.Mapping[str, t.Any]]:
        ...

    def match(
        self,
        path_info: t.Optional[str] = None,
        method: t.Optional[str] = None,
        return_rule: bool = False,
        query_args: t.Optional[t.Union[t.Mapping[str, t.Any], str]] = None,
        websocket: t.Optional[bool] = None,
    ) -> t.Tuple[t.Union[str, Rule], t.Mapping[str, t.Any]]:
        """The usage is simple: you just pass the match method the current
        path info as well as the method (which defaults to `GET`).  The
        following things can then happen:

        - you receive a `NotFound` exception that indicates that no URL is
          matching.  A `NotFound` exception is also a WSGI application you
          can call to get a default page not found page (happens to be the
          same object as `werkzeug.exceptions.NotFound`)

        - you receive a `MethodNotAllowed` exception that indicates that there
          is a match for this URL but not for the current request method.
          This is useful for RESTful applications.

        - you receive a `RequestRedirect` exception with a `new_url`
          attribute.  This exception is used to notify you about a request
          Werkzeug requests from your WSGI application.  This is for example the
          case if you request ``/foo`` although the correct URL is ``/foo/``
          You can use the `RequestRedirect` instance as response-like object
          similar to all other subclasses of `HTTPException`.

        - you receive a ``WebsocketMismatch`` exception if the only
          match is a WebSocket rule but the bind is an HTTP request, or
          if the match is an HTTP rule but the bind is a WebSocket
          request.

        - you get a tuple in the form ``(endpoint, arguments)`` if there is
          a match (unless `return_rule` is True, in which case you get a tuple
          in the form ``(rule, arguments)``)

        If the path info is not passed to the match method the default path
        info of the map is used (defaults to the root URL if not defined
        explicitly).

        All of the exceptions raised are subclasses of `HTTPException` so they
        can be used as WSGI responses. They will all render generic error or
        redirect pages.

        Here is a small example for matching:

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.match("/", "GET")
        ('index', {})
        >>> urls.match("/downloads/42")
        ('downloads/show', {'id': 42})

        And here is what happens on redirect and missing URLs:

        >>> urls.match("/downloads")
        Traceback (most recent call last):
          ...
        RequestRedirect: http://example.com/downloads/
        >>> urls.match("/missing")
        Traceback (most recent call last):
          ...
        NotFound: 404 Not Found

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        :param return_rule: return the rule that matched instead of just the
                            endpoint (defaults to `False`).
        :param query_args: optional query arguments that are used for
                           automatic redirects as string or dictionary.  It's
                           currently not possible to use the query arguments
                           for URL matching.
        :param websocket: Match WebSocket instead of HTTP requests. A
            websocket request has a ``ws`` or ``wss``
            :attr:`url_scheme`. This overrides that detection.

        .. versionadded:: 1.0
            Added ``websocket``.

        .. versionchanged:: 0.8
            ``query_args`` can be a string.

        .. versionadded:: 0.7
            Added ``query_args``.

        .. versionadded:: 0.6
            Added ``return_rule``.
        """
        self.map.update()
        if path_info is None:
            path_info = self.path_info
        else:
            path_info = _to_str(path_info, self.map.charset)
        if query_args is None:
            query_args = self.query_args or {}
        method = (method or self.default_method).upper()

        if websocket is None:
            websocket = self.websocket

        require_redirect = False

        domain_part = self.server_name if self.map.host_matching else self.subdomain
        path_part = f"/{path_info.lstrip('/')}" if path_info else ""
        path = f"{domain_part}|{path_part}"

        have_match_for = set()
        websocket_mismatch = False

        for rule in self.map._rules:
            try:
                rv = rule.match(path, method)
            except RequestPath as e:
                raise RequestRedirect(
                    self.make_redirect_url(
                        url_quote(e.path_info, self.map.charset, safe="/:|+"),
                        query_args,
                    )
                ) from None
            except RequestAliasRedirect as e:
                raise RequestRedirect(
                    self.make_alias_redirect_url(
                        path, rule.endpoint, e.matched_values, method, query_args
                    )
                ) from None
            if rv is None:
                continue
            if rule.methods is not None and method not in rule.methods:
                have_match_for.update(rule.methods)
                continue

            if rule.websocket != websocket:
                websocket_mismatch = True
                continue

            if self.map.redirect_defaults:
                redirect_url = self.get_default_redirect(rule, method, rv, query_args)
                if redirect_url is not None:
                    raise RequestRedirect(redirect_url)

            if rule.redirect_to is not None:
                if isinstance(rule.redirect_to, str):

                    def _handle_match(match: t.Match[str]) -> str:
                        value = rv[match.group(1)]  # type: ignore
                        return rule._converters[match.group(1)].to_url(value)

                    redirect_url = _simple_rule_re.sub(_handle_match, rule.redirect_to)
                else:
                    redirect_url = rule.redirect_to(self, **rv)

                if self.subdomain:
                    netloc = f"{self.subdomain}.{self.server_name}"
                else:
                    netloc = self.server_name

                raise RequestRedirect(
                    url_join(
                        f"{self.url_scheme or 'http'}://{netloc}{self.script_name}",
                        redirect_url,
                    )
                )

            if require_redirect:
                raise RequestRedirect(
                    self.make_redirect_url(
                        url_quote(path_info, self.map.charset, safe="/:|+"), query_args
                    )
                )

            if return_rule:
                return rule, rv
            else:
                return rule.endpoint, rv

        if have_match_for:
            raise MethodNotAllowed(valid_methods=list(have_match_for))

        if websocket_mismatch:
            raise WebsocketMismatch()

        raise NotFound()

    def test(
        self, path_info: t.Optional[str] = None, method: t.Optional[str] = None
    ) -> bool:
        """Test if a rule would match.  Works like `match` but returns `True`
        if the URL matches, or `False` if it does not exist.

        :param path_info: the path info to use for matching.  Overrides the
                          path info specified on binding.
        :param method: the HTTP method used for matching.  Overrides the
                       method specified on binding.
        """
        try:
            self.match(path_info, method)
        except RequestRedirect:
            pass
        except HTTPException:
            return False
        return True

    def allowed_methods(self, path_info: t.Optional[str] = None) -> t.Iterable[str]:
        """Returns the valid methods that match for a given path.

        .. versionadded:: 0.7
        """
        try:
            self.match(path_info, method="--")
        except MethodNotAllowed as e:
            return e.valid_methods  # type: ignore
        except HTTPException:
            pass
        return []

    def get_host(self, domain_part: t.Optional[str]) -> str:
        """Figures out the full host name for the given domain part.  The
        domain part is a subdomain in case host matching is disabled or
        a full host name.
        """
        if self.map.host_matching:
            if domain_part is None:
                return self.server_name
            return _to_str(domain_part, "ascii")
        subdomain = domain_part
        if subdomain is None:
            subdomain = self.subdomain
        else:
            subdomain = _to_str(subdomain, "ascii")

        if subdomain:
            return f"{subdomain}.{self.server_name}"
        else:
            return self.server_name

    def get_default_redirect(
        self,
        rule: Rule,
        method: str,
        values: t.MutableMapping[str, t.Any],
        query_args: t.Union[t.Mapping[str, t.Any], str],
    ) -> t.Optional[str]:
        """A helper that returns the URL to redirect to if it finds one.
        This is used for default redirecting only.

        :internal:
        """
        assert self.map.redirect_defaults
        for r in self.map._rules_by_endpoint[rule.endpoint]:
            # every rule that comes after this one, including ourself
            # has a lower priority for the defaults.  We order the ones
            # with the highest priority up for building.
            if r is rule:
                break
            if r.provides_defaults_for(rule) and r.suitable_for(values, method):
                values.update(r.defaults)  # type: ignore
                domain_part, path = r.build(values)  # type: ignore
                return self.make_redirect_url(path, query_args, domain_part=domain_part)
        return None

    def encode_query_args(self, query_args: t.Union[t.Mapping[str, t.Any], str]) -> str:
        if not isinstance(query_args, str):
            return url_encode(query_args, self.map.charset)
        return query_args

    def make_redirect_url(
        self,
        path_info: str,
        query_args: t.Optional[t.Union[t.Mapping[str, t.Any], str]] = None,
        domain_part: t.Optional[str] = None,
    ) -> str:
        """Creates a redirect URL.

        :internal:
        """
        if query_args:
            suffix = f"?{self.encode_query_args(query_args)}"
        else:
            suffix = ""

        scheme = self.url_scheme or "http"
        host = self.get_host(domain_part)
        path = posixpath.join(self.script_name.strip("/"), path_info.lstrip("/"))
        return f"{scheme}://{host}/{path}{suffix}"

    def make_alias_redirect_url(
        self,
        path: str,
        endpoint: str,
        values: t.Mapping[str, t.Any],
        method: str,
        query_args: t.Union[t.Mapping[str, t.Any], str],
    ) -> str:
        """Internally called to make an alias redirect URL."""
        url = self.build(
            endpoint, values, method, append_unknown=False, force_external=True
        )
        if query_args:
            url += f"?{self.encode_query_args(query_args)}"
        assert url != path, "detected invalid alias setting. No canonical URL found"
        return url

    def _partial_build(
        self,
        endpoint: str,
        values: t.Mapping[str, t.Any],
        method: t.Optional[str],
        append_unknown: bool,
    ) -> t.Optional[t.Tuple[str, str, bool]]:
        """Helper for :meth:`build`.  Returns subdomain and path for the
        rule that accepts this endpoint, values and method.

        :internal:
        """
        # in case the method is none, try with the default method first
        if method is None:
            rv = self._partial_build(
                endpoint, values, self.default_method, append_unknown
            )
            if rv is not None:
                return rv

        # Default method did not match or a specific method is passed.
        # Check all for first match with matching host. If no matching
        # host is found, go with first result.
        first_match = None

        for rule in self.map._rules_by_endpoint.get(endpoint, ()):
            if rule.suitable_for(values, method):
                build_rv = rule.build(values, append_unknown)

                if build_rv is not None:
                    rv = (build_rv[0], build_rv[1], rule.websocket)
                    if self.map.host_matching:
                        if rv[0] == self.server_name:
                            return rv
                        elif first_match is None:
                            first_match = rv
                    else:
                        return rv

        return first_match

    def build(
        self,
        endpoint: str,
        values: t.Optional[t.Mapping[str, t.Any]] = None,
        method: t.Optional[str] = None,
        force_external: bool = False,
        append_unknown: bool = True,
        url_scheme: t.Optional[str] = None,
    ) -> str:
        """Building URLs works pretty much the other way round.  Instead of
        `match` you call `build` and pass it the endpoint and a dict of
        arguments for the placeholders.

        The `build` function also accepts an argument called `force_external`
        which, if you set it to `True` will force external URLs. Per default
        external URLs (include the server name) will only be used if the
        target URL is on a different subdomain.

        >>> m = Map([
        ...     Rule('/', endpoint='index'),
        ...     Rule('/downloads/', endpoint='downloads/index'),
        ...     Rule('/downloads/<int:id>', endpoint='downloads/show')
        ... ])
        >>> urls = m.bind("example.com", "/")
        >>> urls.build("index", {})
        '/'
        >>> urls.build("downloads/show", {'id': 42})
        '/downloads/42'
        >>> urls.build("downloads/show", {'id': 42}, force_external=True)
        'http://example.com/downloads/42'

        Because URLs cannot contain non ASCII data you will always get
        bytes back.  Non ASCII characters are urlencoded with the
        charset defined on the map instance.

        Additional values are converted to strings and appended to the URL as
        URL querystring parameters:

        >>> urls.build("index", {'q': 'My Searchstring'})
        '/?q=My+Searchstring'

        When processing those additional values, lists are furthermore
        interpreted as multiple values (as per
        :py:class:`werkzeug.datastructures.MultiDict`):

        >>> urls.build("index", {'q': ['a', 'b', 'c']})
        '/?q=a&q=b&q=c'

        Passing a ``MultiDict`` will also add multiple values:

        >>> urls.build("index", MultiDict((('p', 'z'), ('q', 'a'), ('q', 'b'))))
        '/?p=z&q=a&q=b'

        If a rule does not exist when building a `BuildError` exception is
        raised.

        The build method accepts an argument called `method` which allows you
        to specify the method you want to have an URL built for if you have
        different methods for the same endpoint specified.

        :param endpoint: the endpoint of the URL to build.
        :param values: the values for the URL to build.  Unhandled values are
                       appended to the URL as query parameters.
        :param method: the HTTP method for the rule if there are different
                       URLs for different methods on the same endpoint.
        :param force_external: enforce full canonical external URLs. If the URL
                               scheme is not provided, this will generate
                               a protocol-relative URL.
        :param append_unknown: unknown parameters are appended to the generated
                               URL as query string argument.  Disable this
                               if you want the builder to ignore those.
        :param url_scheme: Scheme to use in place of the bound
            :attr:`url_scheme`.

        .. versionchanged:: 2.0
            Added the ``url_scheme`` parameter.

        .. versionadded:: 0.6
           Added the ``append_unknown`` parameter.
        """
        self.map.update()

        if values:
            temp_values: t.Dict[str, t.Union[t.List[t.Any], t.Any]] = {}
            always_list = isinstance(values, MultiDict)
            key: str
            value: t.Optional[t.Union[t.List[t.Any], t.Any]]

            # For MultiDict, dict.items(values) is like values.lists()
            # without the call or list coercion overhead.
            for key, value in dict.items(values):  # type: ignore
                if value is None:
                    continue

                if always_list or isinstance(value, (list, tuple)):
                    value = [v for v in value if v is not None]

                    if not value:
                        continue

                    if len(value) == 1:
                        value = value[0]

                temp_values[key] = value

            values = temp_values
        else:
            values = {}

        rv = self._partial_build(endpoint, values, method, append_unknown)
        if rv is None:
            raise BuildError(endpoint, values, method, self)

        domain_part, path, websocket = rv
        host = self.get_host(domain_part)

        if url_scheme is None:
            url_scheme = self.url_scheme

        # Always build WebSocket routes with the scheme (browsers
        # require full URLs). If bound to a WebSocket, ensure that HTTP
        # routes are built with an HTTP scheme.
        secure = url_scheme in {"https", "wss"}

        if websocket:
            force_external = True
            url_scheme = "wss" if secure else "ws"
        elif url_scheme:
            url_scheme = "https" if secure else "http"

        # shortcut this.
        if not force_external and (
            (self.map.host_matching and host == self.server_name)
            or (not self.map.host_matching and domain_part == self.subdomain)
        ):
            return f"{self.script_name.rstrip('/')}/{path.lstrip('/')}"

        scheme = f"{url_scheme}:" if url_scheme else ""
        return f"{scheme}//{host}{self.script_name[:-1]}/{path.lstrip('/')}"
