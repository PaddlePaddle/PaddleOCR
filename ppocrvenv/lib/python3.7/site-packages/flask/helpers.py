import os
import pkgutil
import socket
import sys
import typing as t
import warnings
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
from functools import update_wrapper
from threading import RLock

import werkzeug.utils
from werkzeug.exceptions import NotFound
from werkzeug.routing import BuildError
from werkzeug.urls import url_quote

from .globals import _app_ctx_stack
from .globals import _request_ctx_stack
from .globals import current_app
from .globals import request
from .globals import session
from .signals import message_flashed

if t.TYPE_CHECKING:
    from .wrappers import Response


def get_env() -> str:
    """Get the environment the app is running in, indicated by the
    :envvar:`FLASK_ENV` environment variable. The default is
    ``'production'``.
    """
    return os.environ.get("FLASK_ENV") or "production"


def get_debug_flag() -> bool:
    """Get whether debug mode should be enabled for the app, indicated
    by the :envvar:`FLASK_DEBUG` environment variable. The default is
    ``True`` if :func:`.get_env` returns ``'development'``, or ``False``
    otherwise.
    """
    val = os.environ.get("FLASK_DEBUG")

    if not val:
        return get_env() == "development"

    return val.lower() not in ("0", "false", "no")


def get_load_dotenv(default: bool = True) -> bool:
    """Get whether the user has disabled loading dotenv files by setting
    :envvar:`FLASK_SKIP_DOTENV`. The default is ``True``, load the
    files.

    :param default: What to return if the env var isn't set.
    """
    val = os.environ.get("FLASK_SKIP_DOTENV")

    if not val:
        return default

    return val.lower() in ("0", "false", "no")


def stream_with_context(
    generator_or_function: t.Union[
        t.Iterator[t.AnyStr], t.Callable[..., t.Iterator[t.AnyStr]]
    ]
) -> t.Iterator[t.AnyStr]:
    """Request contexts disappear when the response is started on the server.
    This is done for efficiency reasons and to make it less likely to encounter
    memory leaks with badly written WSGI middlewares.  The downside is that if
    you are using streamed responses, the generator cannot access request bound
    information any more.

    This function however can help you keep the context around for longer::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            @stream_with_context
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(generate())

    Alternatively it can also be used around a specific generator::

        from flask import stream_with_context, request, Response

        @app.route('/stream')
        def streamed_response():
            def generate():
                yield 'Hello '
                yield request.args['name']
                yield '!'
            return Response(stream_with_context(generate()))

    .. versionadded:: 0.9
    """
    try:
        gen = iter(generator_or_function)  # type: ignore
    except TypeError:

        def decorator(*args: t.Any, **kwargs: t.Any) -> t.Any:
            gen = generator_or_function(*args, **kwargs)  # type: ignore
            return stream_with_context(gen)

        return update_wrapper(decorator, generator_or_function)  # type: ignore

    def generator() -> t.Generator:
        ctx = _request_ctx_stack.top
        if ctx is None:
            raise RuntimeError(
                "Attempted to stream with context but "
                "there was no context in the first place to keep around."
            )
        with ctx:
            # Dummy sentinel.  Has to be inside the context block or we're
            # not actually keeping the context around.
            yield None

            # The try/finally is here so that if someone passes a WSGI level
            # iterator in we're still running the cleanup logic.  Generators
            # don't need that because they are closed on their destruction
            # automatically.
            try:
                yield from gen
            finally:
                if hasattr(gen, "close"):
                    gen.close()  # type: ignore

    # The trick is to start the generator.  Then the code execution runs until
    # the first dummy None is yielded at which point the context was already
    # pushed.  This item is discarded.  Then when the iteration continues the
    # real generator is executed.
    wrapped_g = generator()
    next(wrapped_g)
    return wrapped_g


def make_response(*args: t.Any) -> "Response":
    """Sometimes it is necessary to set additional headers in a view.  Because
    views do not have to return response objects but can return a value that
    is converted into a response object by Flask itself, it becomes tricky to
    add headers to it.  This function can be called instead of using a return
    and you will get a response object which you can use to attach headers.

    If view looked like this and you want to add a new header::

        def index():
            return render_template('index.html', foo=42)

    You can now do something like this::

        def index():
            response = make_response(render_template('index.html', foo=42))
            response.headers['X-Parachutes'] = 'parachutes are cool'
            return response

    This function accepts the very same arguments you can return from a
    view function.  This for example creates a response with a 404 error
    code::

        response = make_response(render_template('not_found.html'), 404)

    The other use case of this function is to force the return value of a
    view function into a response which is helpful with view
    decorators::

        response = make_response(view_function())
        response.headers['X-Parachutes'] = 'parachutes are cool'

    Internally this function does the following things:

    -   if no arguments are passed, it creates a new response argument
    -   if one argument is passed, :meth:`flask.Flask.make_response`
        is invoked with it.
    -   if more than one argument is passed, the arguments are passed
        to the :meth:`flask.Flask.make_response` function as tuple.

    .. versionadded:: 0.6
    """
    if not args:
        return current_app.response_class()
    if len(args) == 1:
        args = args[0]
    return current_app.make_response(args)


def url_for(endpoint: str, **values: t.Any) -> str:
    """Generates a URL to the given endpoint with the method provided.

    Variable arguments that are unknown to the target endpoint are appended
    to the generated URL as query arguments.  If the value of a query argument
    is ``None``, the whole pair is skipped.  In case blueprints are active
    you can shortcut references to the same blueprint by prefixing the
    local endpoint with a dot (``.``).

    This will reference the index function local to the current blueprint::

        url_for('.index')

    See :ref:`url-building`.

    Configuration values ``APPLICATION_ROOT`` and ``SERVER_NAME`` are only used when
    generating URLs outside of a request context.

    To integrate applications, :class:`Flask` has a hook to intercept URL build
    errors through :attr:`Flask.url_build_error_handlers`.  The `url_for`
    function results in a :exc:`~werkzeug.routing.BuildError` when the current
    app does not have a URL for the given endpoint and values.  When it does, the
    :data:`~flask.current_app` calls its :attr:`~Flask.url_build_error_handlers` if
    it is not ``None``, which can return a string to use as the result of
    `url_for` (instead of `url_for`'s default to raise the
    :exc:`~werkzeug.routing.BuildError` exception) or re-raise the exception.
    An example::

        def external_url_handler(error, endpoint, values):
            "Looks up an external URL when `url_for` cannot build a URL."
            # This is an example of hooking the build_error_handler.
            # Here, lookup_url is some utility function you've built
            # which looks up the endpoint in some external URL registry.
            url = lookup_url(endpoint, **values)
            if url is None:
                # External lookup did not have a URL.
                # Re-raise the BuildError, in context of original traceback.
                exc_type, exc_value, tb = sys.exc_info()
                if exc_value is error:
                    raise exc_type(exc_value).with_traceback(tb)
                else:
                    raise error
            # url_for will use this result, instead of raising BuildError.
            return url

        app.url_build_error_handlers.append(external_url_handler)

    Here, `error` is the instance of :exc:`~werkzeug.routing.BuildError`, and
    `endpoint` and `values` are the arguments passed into `url_for`.  Note
    that this is for building URLs outside the current application, and not for
    handling 404 NotFound errors.

    .. versionadded:: 0.10
       The `_scheme` parameter was added.

    .. versionadded:: 0.9
       The `_anchor` and `_method` parameters were added.

    .. versionadded:: 0.9
       Calls :meth:`Flask.handle_build_error` on
       :exc:`~werkzeug.routing.BuildError`.

    :param endpoint: the endpoint of the URL (name of the function)
    :param values: the variable arguments of the URL rule
    :param _external: if set to ``True``, an absolute URL is generated. Server
      address can be changed via ``SERVER_NAME`` configuration variable which
      falls back to the `Host` header, then to the IP and port of the request.
    :param _scheme: a string specifying the desired URL scheme. The `_external`
      parameter must be set to ``True`` or a :exc:`ValueError` is raised. The default
      behavior uses the same scheme as the current request, or
      :data:`PREFERRED_URL_SCHEME` if no request context is available.
      This also can be set to an empty string to build protocol-relative
      URLs.
    :param _anchor: if provided this is added as anchor to the URL.
    :param _method: if provided this explicitly specifies an HTTP method.
    """
    appctx = _app_ctx_stack.top
    reqctx = _request_ctx_stack.top

    if appctx is None:
        raise RuntimeError(
            "Attempted to generate a URL without the application context being"
            " pushed. This has to be executed when application context is"
            " available."
        )

    # If request specific information is available we have some extra
    # features that support "relative" URLs.
    if reqctx is not None:
        url_adapter = reqctx.url_adapter
        blueprint_name = request.blueprint

        if endpoint[:1] == ".":
            if blueprint_name is not None:
                endpoint = f"{blueprint_name}{endpoint}"
            else:
                endpoint = endpoint[1:]

        external = values.pop("_external", False)

    # Otherwise go with the url adapter from the appctx and make
    # the URLs external by default.
    else:
        url_adapter = appctx.url_adapter

        if url_adapter is None:
            raise RuntimeError(
                "Application was not able to create a URL adapter for request"
                " independent URL generation. You might be able to fix this by"
                " setting the SERVER_NAME config variable."
            )

        external = values.pop("_external", True)

    anchor = values.pop("_anchor", None)
    method = values.pop("_method", None)
    scheme = values.pop("_scheme", None)
    appctx.app.inject_url_defaults(endpoint, values)

    # This is not the best way to deal with this but currently the
    # underlying Werkzeug router does not support overriding the scheme on
    # a per build call basis.
    old_scheme = None
    if scheme is not None:
        if not external:
            raise ValueError("When specifying _scheme, _external must be True")
        old_scheme = url_adapter.url_scheme
        url_adapter.url_scheme = scheme

    try:
        try:
            rv = url_adapter.build(
                endpoint, values, method=method, force_external=external
            )
        finally:
            if old_scheme is not None:
                url_adapter.url_scheme = old_scheme
    except BuildError as error:
        # We need to inject the values again so that the app callback can
        # deal with that sort of stuff.
        values["_external"] = external
        values["_anchor"] = anchor
        values["_method"] = method
        values["_scheme"] = scheme
        return appctx.app.handle_url_build_error(error, endpoint, values)

    if anchor is not None:
        rv += f"#{url_quote(anchor)}"
    return rv


def get_template_attribute(template_name: str, attribute: str) -> t.Any:
    """Loads a macro (or variable) a template exports.  This can be used to
    invoke a macro from within Python code.  If you for example have a
    template named :file:`_cider.html` with the following contents:

    .. sourcecode:: html+jinja

       {% macro hello(name) %}Hello {{ name }}!{% endmacro %}

    You can access this from Python code like this::

        hello = get_template_attribute('_cider.html', 'hello')
        return hello('World')

    .. versionadded:: 0.2

    :param template_name: the name of the template
    :param attribute: the name of the variable of macro to access
    """
    return getattr(current_app.jinja_env.get_template(template_name).module, attribute)


def flash(message: str, category: str = "message") -> None:
    """Flashes a message to the next request.  In order to remove the
    flashed message from the session and to display it to the user,
    the template has to call :func:`get_flashed_messages`.

    .. versionchanged:: 0.3
       `category` parameter added.

    :param message: the message to be flashed.
    :param category: the category for the message.  The following values
                     are recommended: ``'message'`` for any kind of message,
                     ``'error'`` for errors, ``'info'`` for information
                     messages and ``'warning'`` for warnings.  However any
                     kind of string can be used as category.
    """
    # Original implementation:
    #
    #     session.setdefault('_flashes', []).append((category, message))
    #
    # This assumed that changes made to mutable structures in the session are
    # always in sync with the session object, which is not true for session
    # implementations that use external storage for keeping their keys/values.
    flashes = session.get("_flashes", [])
    flashes.append((category, message))
    session["_flashes"] = flashes
    message_flashed.send(
        current_app._get_current_object(),  # type: ignore
        message=message,
        category=category,
    )


def get_flashed_messages(
    with_categories: bool = False, category_filter: t.Iterable[str] = ()
) -> t.Union[t.List[str], t.List[t.Tuple[str, str]]]:
    """Pulls all flashed messages from the session and returns them.
    Further calls in the same request to the function will return
    the same messages.  By default just the messages are returned,
    but when `with_categories` is set to ``True``, the return value will
    be a list of tuples in the form ``(category, message)`` instead.

    Filter the flashed messages to one or more categories by providing those
    categories in `category_filter`.  This allows rendering categories in
    separate html blocks.  The `with_categories` and `category_filter`
    arguments are distinct:

    * `with_categories` controls whether categories are returned with message
      text (``True`` gives a tuple, where ``False`` gives just the message text).
    * `category_filter` filters the messages down to only those matching the
      provided categories.

    See :doc:`/patterns/flashing` for examples.

    .. versionchanged:: 0.3
       `with_categories` parameter added.

    .. versionchanged:: 0.9
        `category_filter` parameter added.

    :param with_categories: set to ``True`` to also receive categories.
    :param category_filter: filter of categories to limit return values.  Only
                            categories in the list will be returned.
    """
    flashes = _request_ctx_stack.top.flashes
    if flashes is None:
        _request_ctx_stack.top.flashes = flashes = (
            session.pop("_flashes") if "_flashes" in session else []
        )
    if category_filter:
        flashes = list(filter(lambda f: f[0] in category_filter, flashes))
    if not with_categories:
        return [x[1] for x in flashes]
    return flashes


def _prepare_send_file_kwargs(
    download_name: t.Optional[str] = None,
    attachment_filename: t.Optional[str] = None,
    etag: t.Optional[t.Union[bool, str]] = None,
    add_etags: t.Optional[t.Union[bool]] = None,
    max_age: t.Optional[
        t.Union[int, t.Callable[[t.Optional[str]], t.Optional[int]]]
    ] = None,
    cache_timeout: t.Optional[int] = None,
    **kwargs: t.Any,
) -> t.Dict[str, t.Any]:
    if attachment_filename is not None:
        warnings.warn(
            "The 'attachment_filename' parameter has been renamed to"
            " 'download_name'. The old name will be removed in Flask"
            " 2.1.",
            DeprecationWarning,
            stacklevel=3,
        )
        download_name = attachment_filename

    if cache_timeout is not None:
        warnings.warn(
            "The 'cache_timeout' parameter has been renamed to"
            " 'max_age'. The old name will be removed in Flask 2.1.",
            DeprecationWarning,
            stacklevel=3,
        )
        max_age = cache_timeout

    if add_etags is not None:
        warnings.warn(
            "The 'add_etags' parameter has been renamed to 'etag'. The"
            " old name will be removed in Flask 2.1.",
            DeprecationWarning,
            stacklevel=3,
        )
        etag = add_etags

    if max_age is None:
        max_age = current_app.get_send_file_max_age

    kwargs.update(
        environ=request.environ,
        download_name=download_name,
        etag=etag,
        max_age=max_age,
        use_x_sendfile=current_app.use_x_sendfile,
        response_class=current_app.response_class,
        _root_path=current_app.root_path,  # type: ignore
    )
    return kwargs


def send_file(
    path_or_file: t.Union[os.PathLike, str, t.BinaryIO],
    mimetype: t.Optional[str] = None,
    as_attachment: bool = False,
    download_name: t.Optional[str] = None,
    attachment_filename: t.Optional[str] = None,
    conditional: bool = True,
    etag: t.Union[bool, str] = True,
    add_etags: t.Optional[bool] = None,
    last_modified: t.Optional[t.Union[datetime, int, float]] = None,
    max_age: t.Optional[
        t.Union[int, t.Callable[[t.Optional[str]], t.Optional[int]]]
    ] = None,
    cache_timeout: t.Optional[int] = None,
):
    """Send the contents of a file to the client.

    The first argument can be a file path or a file-like object. Paths
    are preferred in most cases because Werkzeug can manage the file and
    get extra information from the path. Passing a file-like object
    requires that the file is opened in binary mode, and is mostly
    useful when building a file in memory with :class:`io.BytesIO`.

    Never pass file paths provided by a user. The path is assumed to be
    trusted, so a user could craft a path to access a file you didn't
    intend. Use :func:`send_from_directory` to safely serve
    user-requested paths from within a directory.

    If the WSGI server sets a ``file_wrapper`` in ``environ``, it is
    used, otherwise Werkzeug's built-in wrapper is used. Alternatively,
    if the HTTP server supports ``X-Sendfile``, configuring Flask with
    ``USE_X_SENDFILE = True`` will tell the server to send the given
    path, which is much more efficient than reading it in Python.

    :param path_or_file: The path to the file to send, relative to the
        current working directory if a relative path is given.
        Alternatively, a file-like object opened in binary mode. Make
        sure the file pointer is seeked to the start of the data.
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

    .. versionchanged:: 2.0
        ``download_name`` replaces the ``attachment_filename``
        parameter. If ``as_attachment=False``, it is passed with
        ``Content-Disposition: inline`` instead.

    .. versionchanged:: 2.0
        ``max_age`` replaces the ``cache_timeout`` parameter.
        ``conditional`` is enabled and ``max_age`` is not set by
        default.

    .. versionchanged:: 2.0
        ``etag`` replaces the ``add_etags`` parameter. It can be a
        string to use instead of generating one.

    .. versionchanged:: 2.0
        Passing a file-like object that inherits from
        :class:`~io.TextIOBase` will raise a :exc:`ValueError` rather
        than sending an empty file.

    .. versionadded:: 2.0
        Moved the implementation to Werkzeug. This is now a wrapper to
        pass some Flask-specific arguments.

    .. versionchanged:: 1.1
        ``filename`` may be a :class:`~os.PathLike` object.

    .. versionchanged:: 1.1
        Passing a :class:`~io.BytesIO` object supports range requests.

    .. versionchanged:: 1.0.3
        Filenames are encoded with ASCII instead of Latin-1 for broader
        compatibility with WSGI servers.

    .. versionchanged:: 1.0
        UTF-8 filenames as specified in :rfc:`2231` are supported.

    .. versionchanged:: 0.12
        The filename is no longer automatically inferred from file
        objects. If you want to use automatic MIME and etag support,
        pass a filename via ``filename_or_fp`` or
        ``attachment_filename``.

    .. versionchanged:: 0.12
        ``attachment_filename`` is preferred over ``filename`` for MIME
        detection.

    .. versionchanged:: 0.9
        ``cache_timeout`` defaults to
        :meth:`Flask.get_send_file_max_age`.

    .. versionchanged:: 0.7
        MIME guessing and etag support for file-like objects was
        deprecated because it was unreliable. Pass a filename if you are
        able to, otherwise attach an etag yourself.

    .. versionchanged:: 0.5
        The ``add_etags``, ``cache_timeout`` and ``conditional``
        parameters were added. The default behavior is to add etags.

    .. versionadded:: 0.2
    """
    return werkzeug.utils.send_file(
        **_prepare_send_file_kwargs(
            path_or_file=path_or_file,
            environ=request.environ,
            mimetype=mimetype,
            as_attachment=as_attachment,
            download_name=download_name,
            attachment_filename=attachment_filename,
            conditional=conditional,
            etag=etag,
            add_etags=add_etags,
            last_modified=last_modified,
            max_age=max_age,
            cache_timeout=cache_timeout,
        )
    )


def safe_join(directory: str, *pathnames: str) -> str:
    """Safely join zero or more untrusted path components to a base
    directory to avoid escaping the base directory.

    :param directory: The trusted base directory.
    :param pathnames: The untrusted path components relative to the
        base directory.
    :return: A safe path, otherwise ``None``.
    """
    warnings.warn(
        "'flask.helpers.safe_join' is deprecated and will be removed in"
        " Flask 2.1. Use 'werkzeug.utils.safe_join' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = werkzeug.utils.safe_join(directory, *pathnames)

    if path is None:
        raise NotFound()

    return path


def send_from_directory(
    directory: t.Union[os.PathLike, str],
    path: t.Union[os.PathLike, str],
    filename: t.Optional[str] = None,
    **kwargs: t.Any,
) -> "Response":
    """Send a file from within a directory using :func:`send_file`.

    .. code-block:: python

        @app.route("/uploads/<path:name>")
        def download_file(name):
            return send_from_directory(
                app.config['UPLOAD_FOLDER'], name, as_attachment=True
            )

    This is a secure way to serve files from a folder, such as static
    files or uploads. Uses :func:`~werkzeug.security.safe_join` to
    ensure the path coming from the client is not maliciously crafted to
    point outside the specified directory.

    If the final path does not point to an existing regular file,
    raises a 404 :exc:`~werkzeug.exceptions.NotFound` error.

    :param directory: The directory that ``path`` must be located under.
    :param path: The path to the file to send, relative to
        ``directory``.
    :param kwargs: Arguments to pass to :func:`send_file`.

    .. versionchanged:: 2.0
        ``path`` replaces the ``filename`` parameter.

    .. versionadded:: 2.0
        Moved the implementation to Werkzeug. This is now a wrapper to
        pass some Flask-specific arguments.

    .. versionadded:: 0.5
    """
    if filename is not None:
        warnings.warn(
            "The 'filename' parameter has been renamed to 'path'. The"
            " old name will be removed in Flask 2.1.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = filename

    return werkzeug.utils.send_from_directory(  # type: ignore
        directory, path, **_prepare_send_file_kwargs(**kwargs)
    )


def get_root_path(import_name: str) -> str:
    """Find the root path of a package, or the path that contains a
    module. If it cannot be found, returns the current working
    directory.

    Not to be confused with the value returned by :func:`find_package`.

    :meta private:
    """
    # Module already imported and has a file attribute. Use that first.
    mod = sys.modules.get(import_name)

    if mod is not None and hasattr(mod, "__file__"):
        return os.path.dirname(os.path.abspath(mod.__file__))

    # Next attempt: check the loader.
    loader = pkgutil.get_loader(import_name)

    # Loader does not exist or we're referring to an unloaded main
    # module or a main module without path (interactive sessions), go
    # with the current working directory.
    if loader is None or import_name == "__main__":
        return os.getcwd()

    if hasattr(loader, "get_filename"):
        filepath = loader.get_filename(import_name)  # type: ignore
    else:
        # Fall back to imports.
        __import__(import_name)
        mod = sys.modules[import_name]
        filepath = getattr(mod, "__file__", None)

        # If we don't have a file path it might be because it is a
        # namespace package. In this case pick the root path from the
        # first module that is contained in the package.
        if filepath is None:
            raise RuntimeError(
                "No root path can be found for the provided module"
                f" {import_name!r}. This can happen because the module"
                " came from an import hook that does not provide file"
                " name information or because it's a namespace package."
                " In this case the root path needs to be explicitly"
                " provided."
            )

    # filepath is import_name.py for a module, or __init__.py for a package.
    return os.path.dirname(os.path.abspath(filepath))


class locked_cached_property(werkzeug.utils.cached_property):
    """A :func:`property` that is only evaluated once. Like
    :class:`werkzeug.utils.cached_property` except access uses a lock
    for thread safety.

    .. versionchanged:: 2.0
        Inherits from Werkzeug's ``cached_property`` (and ``property``).
    """

    def __init__(
        self,
        fget: t.Callable[[t.Any], t.Any],
        name: t.Optional[str] = None,
        doc: t.Optional[str] = None,
    ) -> None:
        super().__init__(fget, name=name, doc=doc)
        self.lock = RLock()

    def __get__(self, obj: object, type: type = None) -> t.Any:  # type: ignore
        if obj is None:
            return self

        with self.lock:
            return super().__get__(obj, type=type)

    def __set__(self, obj: object, value: t.Any) -> None:
        with self.lock:
            super().__set__(obj, value)

    def __delete__(self, obj: object) -> None:
        with self.lock:
            super().__delete__(obj)


def total_seconds(td: timedelta) -> int:
    """Returns the total seconds from a timedelta object.

    :param timedelta td: the timedelta to be converted in seconds

    :returns: number of seconds
    :rtype: int

    .. deprecated:: 2.0
        Will be removed in Flask 2.1. Use
        :meth:`timedelta.total_seconds` instead.
    """
    warnings.warn(
        "'total_seconds' is deprecated and will be removed in Flask"
        " 2.1. Use 'timedelta.total_seconds' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return td.days * 60 * 60 * 24 + td.seconds


def is_ip(value: str) -> bool:
    """Determine if the given string is an IP address.

    :param value: value to check
    :type value: str

    :return: True if string is an IP address
    :rtype: bool
    """
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            socket.inet_pton(family, value)
        except OSError:
            pass
        else:
            return True

    return False


@lru_cache(maxsize=None)
def _split_blueprint_path(name: str) -> t.List[str]:
    out: t.List[str] = [name]

    if "." in name:
        out.extend(_split_blueprint_path(name.rpartition(".")[0]))

    return out
