import codecs
import inspect
import os
import re
import sys
import sysconfig
import traceback
import typing as t
from html import escape
from tokenize import TokenError
from types import CodeType
from types import TracebackType

from .._internal import _to_str
from ..filesystem import get_filesystem_encoding
from ..utils import cached_property
from .console import Console

_coding_re = re.compile(br"coding[:=]\s*([-\w.]+)")
_line_re = re.compile(br"^(.*?)$", re.MULTILINE)
_funcdef_re = re.compile(r"^(\s*def\s)|(.*(?<!\w)lambda(:|\s))|^(\s*@)")

HEADER = """\
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
  "http://www.w3.org/TR/html4/loose.dtd">
<html>
  <head>
    <title>%(title)s // Werkzeug Debugger</title>
    <link rel="stylesheet" href="?__debugger__=yes&amp;cmd=resource&amp;f=style.css"
        type="text/css">
    <!-- We need to make sure this has a favicon so that the debugger does
         not accidentally trigger a request to /favicon.ico which might
         change the application's state. -->
    <link rel="shortcut icon"
        href="?__debugger__=yes&amp;cmd=resource&amp;f=console.png">
    <script src="?__debugger__=yes&amp;cmd=resource&amp;f=debugger.js"></script>
    <script type="text/javascript">
      var TRACEBACK = %(traceback_id)d,
          CONSOLE_MODE = %(console)s,
          EVALEX = %(evalex)s,
          EVALEX_TRUSTED = %(evalex_trusted)s,
          SECRET = "%(secret)s";
    </script>
  </head>
  <body style="background-color: #fff">
    <div class="debugger">
"""
FOOTER = """\
      <div class="footer">
        Brought to you by <strong class="arthur">DON'T PANIC</strong>, your
        friendly Werkzeug powered traceback interpreter.
      </div>
    </div>

    <div class="pin-prompt">
      <div class="inner">
        <h3>Console Locked</h3>
        <p>
          The console is locked and needs to be unlocked by entering the PIN.
          You can find the PIN printed out on the standard output of your
          shell that runs the server.
        <form>
          <p>PIN:
            <input type=text name=pin size=14>
            <input type=submit name=btn value="Confirm Pin">
        </form>
      </div>
    </div>
  </body>
</html>
"""

PAGE_HTML = (
    HEADER
    + """\
<h1>%(exception_type)s</h1>
<div class="detail">
  <p class="errormsg">%(exception)s</p>
</div>
<h2 class="traceback">Traceback <em>(most recent call last)</em></h2>
%(summary)s
<div class="plain">
    <p>
      This is the Copy/Paste friendly version of the traceback.
    </p>
    <textarea cols="50" rows="10" name="code" readonly>%(plaintext)s</textarea>
</div>
<div class="explanation">
  The debugger caught an exception in your WSGI application.  You can now
  look at the traceback which led to the error.  <span class="nojavascript">
  If you enable JavaScript you can also use additional features such as code
  execution (if the evalex feature is enabled), automatic pasting of the
  exceptions and much more.</span>
</div>
"""
    + FOOTER
    + """
<!--

%(plaintext_cs)s

-->
"""
)

CONSOLE_HTML = (
    HEADER
    + """\
<h1>Interactive Console</h1>
<div class="explanation">
In this console you can execute Python expressions in the context of the
application.  The initial namespace was created by the debugger automatically.
</div>
<div class="console"><div class="inner">The Console requires JavaScript.</div></div>
"""
    + FOOTER
)

SUMMARY_HTML = """\
<div class="%(classes)s">
  %(title)s
  <ul>%(frames)s</ul>
  %(description)s
</div>
"""

FRAME_HTML = """\
<div class="frame" id="frame-%(id)d">
  <h4>File <cite class="filename">"%(filename)s"</cite>,
      line <em class="line">%(lineno)s</em>,
      in <code class="function">%(function_name)s</code></h4>
  <div class="source %(library)s">%(lines)s</div>
</div>
"""

SOURCE_LINE_HTML = """\
<tr class="%(classes)s">
  <td class=lineno>%(lineno)s</td>
  <td>%(code)s</td>
</tr>
"""


def render_console_html(secret: str, evalex_trusted: bool = True) -> str:
    return CONSOLE_HTML % {
        "evalex": "true",
        "evalex_trusted": "true" if evalex_trusted else "false",
        "console": "true",
        "title": "Console",
        "secret": secret,
        "traceback_id": -1,
    }


def get_current_traceback(
    ignore_system_exceptions: bool = False,
    show_hidden_frames: bool = False,
    skip: int = 0,
) -> "Traceback":
    """Get the current exception info as `Traceback` object.  Per default
    calling this method will reraise system exceptions such as generator exit,
    system exit or others.  This behavior can be disabled by passing `False`
    to the function as first parameter.
    """
    info = t.cast(
        t.Tuple[t.Type[BaseException], BaseException, TracebackType], sys.exc_info()
    )
    exc_type, exc_value, tb = info

    if ignore_system_exceptions and exc_type in {
        SystemExit,
        KeyboardInterrupt,
        GeneratorExit,
    }:
        raise
    for _ in range(skip):
        if tb.tb_next is None:
            break
        tb = tb.tb_next
    tb = Traceback(exc_type, exc_value, tb)
    if not show_hidden_frames:
        tb.filter_hidden_frames()
    return tb


class Line:
    """Helper for the source renderer."""

    __slots__ = ("lineno", "code", "in_frame", "current")

    def __init__(self, lineno: int, code: str) -> None:
        self.lineno = lineno
        self.code = code
        self.in_frame = False
        self.current = False

    @property
    def classes(self) -> t.List[str]:
        rv = ["line"]
        if self.in_frame:
            rv.append("in-frame")
        if self.current:
            rv.append("current")
        return rv

    def render(self) -> str:
        return SOURCE_LINE_HTML % {
            "classes": " ".join(self.classes),
            "lineno": self.lineno,
            "code": escape(self.code),
        }


class Traceback:
    """Wraps a traceback."""

    def __init__(
        self,
        exc_type: t.Type[BaseException],
        exc_value: BaseException,
        tb: TracebackType,
    ) -> None:
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.tb = tb

        exception_type = exc_type.__name__
        if exc_type.__module__ not in {"builtins", "__builtin__", "exceptions"}:
            exception_type = f"{exc_type.__module__}.{exception_type}"
        self.exception_type = exception_type

        self.groups = []
        memo = set()
        while True:
            self.groups.append(Group(exc_type, exc_value, tb))
            memo.add(id(exc_value))
            exc_value = exc_value.__cause__ or exc_value.__context__  # type: ignore
            if exc_value is None or id(exc_value) in memo:
                break
            exc_type = type(exc_value)
            tb = exc_value.__traceback__  # type: ignore
        self.groups.reverse()
        self.frames = [frame for group in self.groups for frame in group.frames]

    def filter_hidden_frames(self) -> None:
        """Remove the frames according to the paste spec."""
        for group in self.groups:
            group.filter_hidden_frames()

        self.frames[:] = [frame for group in self.groups for frame in group.frames]

    @property
    def is_syntax_error(self) -> bool:
        """Is it a syntax error?"""
        return isinstance(self.exc_value, SyntaxError)

    @property
    def exception(self) -> str:
        """String representation of the final exception."""
        return self.groups[-1].exception

    def log(self, logfile: t.Optional[t.IO[str]] = None) -> None:
        """Log the ASCII traceback into a file object."""
        if logfile is None:
            logfile = sys.stderr
        tb = f"{self.plaintext.rstrip()}\n"
        logfile.write(tb)

    def render_summary(self, include_title: bool = True) -> str:
        """Render the traceback for the interactive console."""
        title = ""
        classes = ["traceback"]
        if not self.frames:
            classes.append("noframe-traceback")
            frames = []
        else:
            library_frames = sum(frame.is_library for frame in self.frames)
            mark_lib = 0 < library_frames < len(self.frames)
            frames = [group.render(mark_lib=mark_lib) for group in self.groups]

        if include_title:
            if self.is_syntax_error:
                title = "Syntax Error"
            else:
                title = "Traceback <em>(most recent call last)</em>:"

        if self.is_syntax_error:
            description = f"<pre class=syntaxerror>{escape(self.exception)}</pre>"
        else:
            description = f"<blockquote>{escape(self.exception)}</blockquote>"

        return SUMMARY_HTML % {
            "classes": " ".join(classes),
            "title": f"<h3>{title if title else ''}</h3>",
            "frames": "\n".join(frames),
            "description": description,
        }

    def render_full(
        self,
        evalex: bool = False,
        secret: t.Optional[str] = None,
        evalex_trusted: bool = True,
    ) -> str:
        """Render the Full HTML page with the traceback info."""
        exc = escape(self.exception)
        return PAGE_HTML % {
            "evalex": "true" if evalex else "false",
            "evalex_trusted": "true" if evalex_trusted else "false",
            "console": "false",
            "title": exc,
            "exception": exc,
            "exception_type": escape(self.exception_type),
            "summary": self.render_summary(include_title=False),
            "plaintext": escape(self.plaintext),
            "plaintext_cs": re.sub("-{2,}", "-", self.plaintext),
            "traceback_id": self.id,
            "secret": secret,
        }

    @cached_property
    def plaintext(self) -> str:
        return "\n".join([group.render_text() for group in self.groups])

    @property
    def id(self) -> int:
        return id(self)


class Group:
    """A group of frames for an exception in a traceback. If the
    exception has a ``__cause__`` or ``__context__``, there are multiple
    exception groups.
    """

    def __init__(
        self,
        exc_type: t.Type[BaseException],
        exc_value: BaseException,
        tb: TracebackType,
    ) -> None:
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.info = None
        if exc_value.__cause__ is not None:
            self.info = (
                "The above exception was the direct cause of the following exception"
            )
        elif exc_value.__context__ is not None:
            self.info = (
                "During handling of the above exception, another exception occurred"
            )

        self.frames = []
        while tb is not None:
            self.frames.append(Frame(exc_type, exc_value, tb))
            tb = tb.tb_next  # type: ignore

    def filter_hidden_frames(self) -> None:
        # An exception may not have a traceback to filter frames, such
        # as one re-raised from ProcessPoolExecutor.
        if not self.frames:
            return

        new_frames: t.List[Frame] = []
        hidden = False

        for frame in self.frames:
            hide = frame.hide
            if hide in ("before", "before_and_this"):
                new_frames = []
                hidden = False
                if hide == "before_and_this":
                    continue
            elif hide in ("reset", "reset_and_this"):
                hidden = False
                if hide == "reset_and_this":
                    continue
            elif hide in ("after", "after_and_this"):
                hidden = True
                if hide == "after_and_this":
                    continue
            elif hide or hidden:
                continue
            new_frames.append(frame)

        # if we only have one frame and that frame is from the codeop
        # module, remove it.
        if len(new_frames) == 1 and self.frames[0].module == "codeop":
            del self.frames[:]

        # if the last frame is missing something went terrible wrong :(
        elif self.frames[-1] in new_frames:
            self.frames[:] = new_frames

    @property
    def exception(self) -> str:
        """String representation of the exception."""
        buf = traceback.format_exception_only(self.exc_type, self.exc_value)
        rv = "".join(buf).strip()
        return _to_str(rv, "utf-8", "replace")

    def render(self, mark_lib: bool = True) -> str:
        out = []
        if self.info is not None:
            out.append(f'<li><div class="exc-divider">{self.info}:</div>')
        for frame in self.frames:
            title = f' title="{escape(frame.info)}"' if frame.info else ""
            out.append(f"<li{title}>{frame.render(mark_lib=mark_lib)}")
        return "\n".join(out)

    def render_text(self) -> str:
        out = []
        if self.info is not None:
            out.append(f"\n{self.info}:\n")
        out.append("Traceback (most recent call last):")
        for frame in self.frames:
            out.append(frame.render_text())
        out.append(self.exception)
        return "\n".join(out)


class Frame:
    """A single frame in a traceback."""

    def __init__(
        self,
        exc_type: t.Type[BaseException],
        exc_value: BaseException,
        tb: TracebackType,
    ) -> None:
        self.lineno = tb.tb_lineno
        self.function_name = tb.tb_frame.f_code.co_name
        self.locals = tb.tb_frame.f_locals
        self.globals = tb.tb_frame.f_globals

        fn = inspect.getsourcefile(tb) or inspect.getfile(tb)
        if fn[-4:] in (".pyo", ".pyc"):
            fn = fn[:-1]
        # if it's a file on the file system resolve the real filename.
        if os.path.isfile(fn):
            fn = os.path.realpath(fn)
        self.filename = _to_str(fn, get_filesystem_encoding())
        self.module = self.globals.get("__name__", self.locals.get("__name__"))
        self.loader = self.globals.get("__loader__", self.locals.get("__loader__"))
        self.code = tb.tb_frame.f_code

        # support for paste's traceback extensions
        self.hide = self.locals.get("__traceback_hide__", False)
        info = self.locals.get("__traceback_info__")
        if info is not None:
            info = _to_str(info, "utf-8", "replace")
        self.info = info

    def render(self, mark_lib: bool = True) -> str:
        """Render a single frame in a traceback."""
        return FRAME_HTML % {
            "id": self.id,
            "filename": escape(self.filename),
            "lineno": self.lineno,
            "function_name": escape(self.function_name),
            "lines": self.render_line_context(),
            "library": "library" if mark_lib and self.is_library else "",
        }

    @cached_property
    def is_library(self) -> bool:
        return any(
            self.filename.startswith(os.path.realpath(path))
            for path in sysconfig.get_paths().values()
        )

    def render_text(self) -> str:
        return (
            f'  File "{self.filename}", line {self.lineno}, in {self.function_name}\n'
            f"    {self.current_line.strip()}"
        )

    def render_line_context(self) -> str:
        before, current, after = self.get_context_lines()
        rv = []

        def render_line(line: str, cls: str) -> None:
            line = line.expandtabs().rstrip()
            stripped_line = line.strip()
            prefix = len(line) - len(stripped_line)
            rv.append(
                f'<pre class="line {cls}"><span class="ws">{" " * prefix}</span>'
                f"{escape(stripped_line) if stripped_line else ' '}</pre>"
            )

        for line in before:
            render_line(line, "before")
        render_line(current, "current")
        for line in after:
            render_line(line, "after")

        return "\n".join(rv)

    def get_annotated_lines(self) -> t.List[Line]:
        """Helper function that returns lines with extra information."""
        lines = [Line(idx + 1, x) for idx, x in enumerate(self.sourcelines)]

        # find function definition and mark lines
        if hasattr(self.code, "co_firstlineno"):
            lineno = self.code.co_firstlineno - 1
            while lineno > 0:
                if _funcdef_re.match(lines[lineno].code):
                    break
                lineno -= 1
            try:
                offset = len(inspect.getblock([f"{x.code}\n" for x in lines[lineno:]]))
            except TokenError:
                offset = 0
            for line in lines[lineno : lineno + offset]:
                line.in_frame = True

        # mark current line
        try:
            lines[self.lineno - 1].current = True
        except IndexError:
            pass

        return lines

    def eval(self, code: t.Union[str, CodeType], mode: str = "single") -> t.Any:
        """Evaluate code in the context of the frame."""
        if isinstance(code, str):
            code = compile(code, "<interactive>", mode)
        return eval(code, self.globals, self.locals)

    @cached_property
    def sourcelines(self) -> t.List[str]:
        """The sourcecode of the file as list of strings."""
        # get sourcecode from loader or file
        source = None
        if self.loader is not None:
            try:
                if hasattr(self.loader, "get_source"):
                    source = self.loader.get_source(self.module)
                elif hasattr(self.loader, "get_source_by_code"):
                    source = self.loader.get_source_by_code(self.code)
            except Exception:
                # we munch the exception so that we don't cause troubles
                # if the loader is broken.
                pass

        if source is None:
            try:
                with open(self.filename, mode="rb") as f:
                    source = f.read()
            except OSError:
                return []

        # already str?  return right away
        if isinstance(source, str):
            return source.splitlines()

        charset = "utf-8"
        if source.startswith(codecs.BOM_UTF8):
            source = source[3:]
        else:
            for idx, match in enumerate(_line_re.finditer(source)):
                coding_match = _coding_re.search(match.group())
                if coding_match is not None:
                    charset = coding_match.group(1).decode("utf-8")
                    break
                if idx > 1:
                    break

        # on broken cookies we fall back to utf-8 too
        charset = _to_str(charset)
        try:
            codecs.lookup(charset)
        except LookupError:
            charset = "utf-8"

        return source.decode(charset, "replace").splitlines()

    def get_context_lines(
        self, context: int = 5
    ) -> t.Tuple[t.List[str], str, t.List[str]]:
        before = self.sourcelines[self.lineno - context - 1 : self.lineno - 1]
        past = self.sourcelines[self.lineno : self.lineno + context]
        return (before, self.current_line, past)

    @property
    def current_line(self) -> str:
        try:
            return self.sourcelines[self.lineno - 1]
        except IndexError:
            return ""

    @cached_property
    def console(self) -> Console:
        return Console(self.globals, self.locals)

    @property
    def id(self) -> int:
        return id(self)
