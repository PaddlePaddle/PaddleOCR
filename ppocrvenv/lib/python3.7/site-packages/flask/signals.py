import typing as t

try:
    from blinker import Namespace

    signals_available = True
except ImportError:
    signals_available = False

    class Namespace:  # type: ignore
        def signal(self, name: str, doc: t.Optional[str] = None) -> "_FakeSignal":
            return _FakeSignal(name, doc)

    class _FakeSignal:
        """If blinker is unavailable, create a fake class with the same
        interface that allows sending of signals but will fail with an
        error on anything else.  Instead of doing anything on send, it
        will just ignore the arguments and do nothing instead.
        """

        def __init__(self, name: str, doc: t.Optional[str] = None) -> None:
            self.name = name
            self.__doc__ = doc

        def send(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
            pass

        def _fail(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
            raise RuntimeError(
                "Signalling support is unavailable because the blinker"
                " library is not installed."
            ) from None

        connect = connect_via = connected_to = temporarily_connected_to = _fail
        disconnect = _fail
        has_receivers_for = receivers_for = _fail
        del _fail


# The namespace for code signals.  If you are not Flask code, do
# not put signals in here.  Create your own namespace instead.
_signals = Namespace()


# Core signals.  For usage examples grep the source code or consult
# the API documentation in docs/api.rst as well as docs/signals.rst
template_rendered = _signals.signal("template-rendered")
before_render_template = _signals.signal("before-render-template")
request_started = _signals.signal("request-started")
request_finished = _signals.signal("request-finished")
request_tearing_down = _signals.signal("request-tearing-down")
got_request_exception = _signals.signal("got-request-exception")
appcontext_tearing_down = _signals.signal("appcontext-tearing-down")
appcontext_pushed = _signals.signal("appcontext-pushed")
appcontext_popped = _signals.signal("appcontext-popped")
message_flashed = _signals.signal("message-flashed")
