"""
Hook wrapper "result" utilities.
"""
import sys


def _raise_wrapfail(wrap_controller, msg):
    co = wrap_controller.gi_code
    raise RuntimeError(
        "wrap_controller at %r %s:%d %s"
        % (co.co_name, co.co_filename, co.co_firstlineno, msg)
    )


class HookCallError(Exception):
    """Hook was called wrongly."""


class _Result:
    def __init__(self, result, excinfo):
        self._result = result
        self._excinfo = excinfo

    @property
    def excinfo(self):
        return self._excinfo

    @classmethod
    def from_call(cls, func):
        __tracebackhide__ = True
        result = excinfo = None
        try:
            result = func()
        except BaseException:
            excinfo = sys.exc_info()

        return cls(result, excinfo)

    def force_result(self, result):
        """Force the result(s) to ``result``.

        If the hook was marked as a ``firstresult`` a single value should
        be set otherwise set a (modified) list of results. Any exceptions
        found during invocation will be deleted.
        """
        self._result = result
        self._excinfo = None

    def get_result(self):
        """Get the result(s) for this hook call.

        If the hook was marked as a ``firstresult`` only a single value
        will be returned otherwise a list of results.
        """
        __tracebackhide__ = True
        if self._excinfo is None:
            return self._result
        else:
            ex = self._excinfo
            raise ex[1].with_traceback(ex[2])
