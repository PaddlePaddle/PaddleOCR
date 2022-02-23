import typing as t
from functools import partial

from werkzeug.local import LocalProxy
from werkzeug.local import LocalStack

if t.TYPE_CHECKING:
    from .app import Flask
    from .ctx import _AppCtxGlobals
    from .sessions import SessionMixin
    from .wrappers import Request

_request_ctx_err_msg = """\
Working outside of request context.

This typically means that you attempted to use functionality that needed
an active HTTP request.  Consult the documentation on testing for
information about how to avoid this problem.\
"""
_app_ctx_err_msg = """\
Working outside of application context.

This typically means that you attempted to use functionality that needed
to interface with the current application object in some way. To solve
this, set up an application context with app.app_context().  See the
documentation for more information.\
"""


def _lookup_req_object(name):
    top = _request_ctx_stack.top
    if top is None:
        raise RuntimeError(_request_ctx_err_msg)
    return getattr(top, name)


def _lookup_app_object(name):
    top = _app_ctx_stack.top
    if top is None:
        raise RuntimeError(_app_ctx_err_msg)
    return getattr(top, name)


def _find_app():
    top = _app_ctx_stack.top
    if top is None:
        raise RuntimeError(_app_ctx_err_msg)
    return top.app


# context locals
_request_ctx_stack = LocalStack()
_app_ctx_stack = LocalStack()
current_app: "Flask" = LocalProxy(_find_app)  # type: ignore
request: "Request" = LocalProxy(partial(_lookup_req_object, "request"))  # type: ignore
session: "SessionMixin" = LocalProxy(  # type: ignore
    partial(_lookup_req_object, "session")
)
g: "_AppCtxGlobals" = LocalProxy(partial(_lookup_app_object, "g"))  # type: ignore
