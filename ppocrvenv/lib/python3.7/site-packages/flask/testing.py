import typing as t
from contextlib import contextmanager
from copy import copy
from types import TracebackType

import werkzeug.test
from click.testing import CliRunner
from werkzeug.test import Client
from werkzeug.urls import url_parse
from werkzeug.wrappers import Request as BaseRequest

from . import _request_ctx_stack
from .cli import ScriptInfo
from .json import dumps as json_dumps
from .sessions import SessionMixin

if t.TYPE_CHECKING:
    from .app import Flask
    from .wrappers import Response


class EnvironBuilder(werkzeug.test.EnvironBuilder):
    """An :class:`~werkzeug.test.EnvironBuilder`, that takes defaults from the
    application.

    :param app: The Flask application to configure the environment from.
    :param path: URL path being requested.
    :param base_url: Base URL where the app is being served, which
        ``path`` is relative to. If not given, built from
        :data:`PREFERRED_URL_SCHEME`, ``subdomain``,
        :data:`SERVER_NAME`, and :data:`APPLICATION_ROOT`.
    :param subdomain: Subdomain name to append to :data:`SERVER_NAME`.
    :param url_scheme: Scheme to use instead of
        :data:`PREFERRED_URL_SCHEME`.
    :param json: If given, this is serialized as JSON and passed as
        ``data``. Also defaults ``content_type`` to
        ``application/json``.
    :param args: other positional arguments passed to
        :class:`~werkzeug.test.EnvironBuilder`.
    :param kwargs: other keyword arguments passed to
        :class:`~werkzeug.test.EnvironBuilder`.
    """

    def __init__(
        self,
        app: "Flask",
        path: str = "/",
        base_url: t.Optional[str] = None,
        subdomain: t.Optional[str] = None,
        url_scheme: t.Optional[str] = None,
        *args: t.Any,
        **kwargs: t.Any,
    ) -> None:
        assert not (base_url or subdomain or url_scheme) or (
            base_url is not None
        ) != bool(
            subdomain or url_scheme
        ), 'Cannot pass "subdomain" or "url_scheme" with "base_url".'

        if base_url is None:
            http_host = app.config.get("SERVER_NAME") or "localhost"
            app_root = app.config["APPLICATION_ROOT"]

            if subdomain:
                http_host = f"{subdomain}.{http_host}"

            if url_scheme is None:
                url_scheme = app.config["PREFERRED_URL_SCHEME"]

            url = url_parse(path)
            base_url = (
                f"{url.scheme or url_scheme}://{url.netloc or http_host}"
                f"/{app_root.lstrip('/')}"
            )
            path = url.path

            if url.query:
                sep = b"?" if isinstance(url.query, bytes) else "?"
                path += sep + url.query

        self.app = app
        super().__init__(path, base_url, *args, **kwargs)

    def json_dumps(self, obj: t.Any, **kwargs: t.Any) -> str:  # type: ignore
        """Serialize ``obj`` to a JSON-formatted string.

        The serialization will be configured according to the config associated
        with this EnvironBuilder's ``app``.
        """
        kwargs.setdefault("app", self.app)
        return json_dumps(obj, **kwargs)


class FlaskClient(Client):
    """Works like a regular Werkzeug test client but has some knowledge about
    how Flask works to defer the cleanup of the request context stack to the
    end of a ``with`` body when used in a ``with`` statement.  For general
    information about how to use this class refer to
    :class:`werkzeug.test.Client`.

    .. versionchanged:: 0.12
       `app.test_client()` includes preset default environment, which can be
       set after instantiation of the `app.test_client()` object in
       `client.environ_base`.

    Basic usage is outlined in the :doc:`/testing` chapter.
    """

    application: "Flask"
    preserve_context = False

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(*args, **kwargs)
        self.environ_base = {
            "REMOTE_ADDR": "127.0.0.1",
            "HTTP_USER_AGENT": f"werkzeug/{werkzeug.__version__}",
        }

    @contextmanager
    def session_transaction(
        self, *args: t.Any, **kwargs: t.Any
    ) -> t.Generator[SessionMixin, None, None]:
        """When used in combination with a ``with`` statement this opens a
        session transaction.  This can be used to modify the session that
        the test client uses.  Once the ``with`` block is left the session is
        stored back.

        ::

            with client.session_transaction() as session:
                session['value'] = 42

        Internally this is implemented by going through a temporary test
        request context and since session handling could depend on
        request variables this function accepts the same arguments as
        :meth:`~flask.Flask.test_request_context` which are directly
        passed through.
        """
        if self.cookie_jar is None:
            raise RuntimeError(
                "Session transactions only make sense with cookies enabled."
            )
        app = self.application
        environ_overrides = kwargs.setdefault("environ_overrides", {})
        self.cookie_jar.inject_wsgi(environ_overrides)
        outer_reqctx = _request_ctx_stack.top
        with app.test_request_context(*args, **kwargs) as c:
            session_interface = app.session_interface
            sess = session_interface.open_session(app, c.request)
            if sess is None:
                raise RuntimeError(
                    "Session backend did not open a session. Check the configuration"
                )

            # Since we have to open a new request context for the session
            # handling we want to make sure that we hide out own context
            # from the caller.  By pushing the original request context
            # (or None) on top of this and popping it we get exactly that
            # behavior.  It's important to not use the push and pop
            # methods of the actual request context object since that would
            # mean that cleanup handlers are called
            _request_ctx_stack.push(outer_reqctx)
            try:
                yield sess
            finally:
                _request_ctx_stack.pop()

            resp = app.response_class()
            if not session_interface.is_null_session(sess):
                session_interface.save_session(app, sess, resp)
            headers = resp.get_wsgi_headers(c.request.environ)
            self.cookie_jar.extract_wsgi(c.request.environ, headers)

    def open(  # type: ignore
        self,
        *args: t.Any,
        as_tuple: bool = False,
        buffered: bool = False,
        follow_redirects: bool = False,
        **kwargs: t.Any,
    ) -> "Response":
        # Same logic as super.open, but apply environ_base and preserve_context.
        request = None

        def copy_environ(other):
            return {
                **self.environ_base,
                **other,
                "flask._preserve_context": self.preserve_context,
            }

        if not kwargs and len(args) == 1:
            arg = args[0]

            if isinstance(arg, werkzeug.test.EnvironBuilder):
                builder = copy(arg)
                builder.environ_base = copy_environ(builder.environ_base or {})
                request = builder.get_request()
            elif isinstance(arg, dict):
                request = EnvironBuilder.from_environ(
                    arg, app=self.application, environ_base=copy_environ({})
                ).get_request()
            elif isinstance(arg, BaseRequest):
                request = copy(arg)
                request.environ = copy_environ(request.environ)

        if request is None:
            kwargs["environ_base"] = copy_environ(kwargs.get("environ_base", {}))
            builder = EnvironBuilder(self.application, *args, **kwargs)

            try:
                request = builder.get_request()
            finally:
                builder.close()

        return super().open(  # type: ignore
            request,
            as_tuple=as_tuple,
            buffered=buffered,
            follow_redirects=follow_redirects,
        )

    def __enter__(self) -> "FlaskClient":
        if self.preserve_context:
            raise RuntimeError("Cannot nest client invocations")
        self.preserve_context = True
        return self

    def __exit__(
        self, exc_type: type, exc_value: BaseException, tb: TracebackType
    ) -> None:
        self.preserve_context = False

        # Normally the request context is preserved until the next
        # request in the same thread comes. When the client exits we
        # want to clean up earlier. Pop request contexts until the stack
        # is empty or a non-preserved one is found.
        while True:
            top = _request_ctx_stack.top

            if top is not None and top.preserved:
                top.pop()
            else:
                break


class FlaskCliRunner(CliRunner):
    """A :class:`~click.testing.CliRunner` for testing a Flask app's
    CLI commands. Typically created using
    :meth:`~flask.Flask.test_cli_runner`. See :ref:`testing-cli`.
    """

    def __init__(self, app: "Flask", **kwargs: t.Any) -> None:
        self.app = app
        super().__init__(**kwargs)

    def invoke(  # type: ignore
        self, cli: t.Any = None, args: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        """Invokes a CLI command in an isolated environment. See
        :meth:`CliRunner.invoke <click.testing.CliRunner.invoke>` for
        full method documentation. See :ref:`testing-cli` for examples.

        If the ``obj`` argument is not given, passes an instance of
        :class:`~flask.cli.ScriptInfo` that knows how to load the Flask
        app being tested.

        :param cli: Command object to invoke. Default is the app's
            :attr:`~flask.app.Flask.cli` group.
        :param args: List of strings to invoke the command with.

        :return: a :class:`~click.testing.Result` object.
        """
        if cli is None:
            cli = self.app.cli

        if "obj" not in kwargs:
            kwargs["obj"] = ScriptInfo(create_app=lambda: self.app)

        return super().invoke(cli, args, **kwargs)
