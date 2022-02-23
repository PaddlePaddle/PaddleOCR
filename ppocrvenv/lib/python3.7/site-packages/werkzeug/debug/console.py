import code
import sys
import typing as t
from html import escape
from types import CodeType

from ..local import Local
from .repr import debug_repr
from .repr import dump
from .repr import helper

if t.TYPE_CHECKING:
    import codeop  # noqa: F401

_local = Local()


class HTMLStringO:
    """A StringO version that HTML escapes on write."""

    def __init__(self) -> None:
        self._buffer: t.List[str] = []

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def seek(self, n: int, mode: int = 0) -> None:
        pass

    def readline(self) -> str:
        if len(self._buffer) == 0:
            return ""
        ret = self._buffer[0]
        del self._buffer[0]
        return ret

    def reset(self) -> str:
        val = "".join(self._buffer)
        del self._buffer[:]
        return val

    def _write(self, x: str) -> None:
        if isinstance(x, bytes):
            x = x.decode("utf-8", "replace")
        self._buffer.append(x)

    def write(self, x: str) -> None:
        self._write(escape(x))

    def writelines(self, x: t.Iterable[str]) -> None:
        self._write(escape("".join(x)))


class ThreadedStream:
    """Thread-local wrapper for sys.stdout for the interactive console."""

    @staticmethod
    def push() -> None:
        if not isinstance(sys.stdout, ThreadedStream):
            sys.stdout = t.cast(t.TextIO, ThreadedStream())
        _local.stream = HTMLStringO()

    @staticmethod
    def fetch() -> str:
        try:
            stream = _local.stream
        except AttributeError:
            return ""
        return stream.reset()  # type: ignore

    @staticmethod
    def displayhook(obj: object) -> None:
        try:
            stream = _local.stream
        except AttributeError:
            return _displayhook(obj)  # type: ignore
        # stream._write bypasses escaping as debug_repr is
        # already generating HTML for us.
        if obj is not None:
            _local._current_ipy.locals["_"] = obj
            stream._write(debug_repr(obj))

    def __setattr__(self, name: str, value: t.Any) -> None:
        raise AttributeError(f"read only attribute {name}")

    def __dir__(self) -> t.List[str]:
        return dir(sys.__stdout__)

    def __getattribute__(self, name: str) -> t.Any:
        try:
            stream = _local.stream
        except AttributeError:
            stream = sys.__stdout__
        return getattr(stream, name)

    def __repr__(self) -> str:
        return repr(sys.__stdout__)


# add the threaded stream as display hook
_displayhook = sys.displayhook
sys.displayhook = ThreadedStream.displayhook


class _ConsoleLoader:
    def __init__(self) -> None:
        self._storage: t.Dict[int, str] = {}

    def register(self, code: CodeType, source: str) -> None:
        self._storage[id(code)] = source
        # register code objects of wrapped functions too.
        for var in code.co_consts:
            if isinstance(var, CodeType):
                self._storage[id(var)] = source

    def get_source_by_code(self, code: CodeType) -> t.Optional[str]:
        try:
            return self._storage[id(code)]
        except KeyError:
            return None


class _InteractiveConsole(code.InteractiveInterpreter):
    locals: t.Dict[str, t.Any]

    def __init__(self, globals: t.Dict[str, t.Any], locals: t.Dict[str, t.Any]) -> None:
        self.loader = _ConsoleLoader()
        locals = {
            **globals,
            **locals,
            "dump": dump,
            "help": helper,
            "__loader__": self.loader,
        }
        super().__init__(locals)
        original_compile = self.compile

        def compile(source: str, filename: str, symbol: str) -> CodeType:
            code = original_compile(source, filename, symbol)
            self.loader.register(code, source)
            return code

        self.compile = compile
        self.more = False
        self.buffer: t.List[str] = []

    def runsource(self, source: str, **kwargs: t.Any) -> str:  # type: ignore
        source = f"{source.rstrip()}\n"
        ThreadedStream.push()
        prompt = "... " if self.more else ">>> "
        try:
            source_to_eval = "".join(self.buffer + [source])
            if super().runsource(source_to_eval, "<debugger>", "single"):
                self.more = True
                self.buffer.append(source)
            else:
                self.more = False
                del self.buffer[:]
        finally:
            output = ThreadedStream.fetch()
        return prompt + escape(source) + output

    def runcode(self, code: CodeType) -> None:
        try:
            exec(code, self.locals)
        except Exception:
            self.showtraceback()

    def showtraceback(self) -> None:
        from .tbtools import get_current_traceback

        tb = get_current_traceback(skip=1)
        sys.stdout._write(tb.render_summary())  # type: ignore

    def showsyntaxerror(self, filename: t.Optional[str] = None) -> None:
        from .tbtools import get_current_traceback

        tb = get_current_traceback(skip=4)
        sys.stdout._write(tb.render_summary())  # type: ignore

    def write(self, data: str) -> None:
        sys.stdout.write(data)


class Console:
    """An interactive console."""

    def __init__(
        self,
        globals: t.Optional[t.Dict[str, t.Any]] = None,
        locals: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> None:
        if locals is None:
            locals = {}
        if globals is None:
            globals = {}
        self._ipy = _InteractiveConsole(globals, locals)

    def eval(self, code: str) -> str:
        _local._current_ipy = self._ipy
        old_sys_stdout = sys.stdout
        try:
            return self._ipy.runsource(code)
        finally:
            sys.stdout = old_sys_stdout
