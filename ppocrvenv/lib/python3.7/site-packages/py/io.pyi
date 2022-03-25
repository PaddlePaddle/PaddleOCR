from io import StringIO as TextIO
from io import BytesIO as BytesIO
from typing import Any, AnyStr, Callable, Generic, IO, List, Optional, Text, Tuple, TypeVar, Union, overload
from typing_extensions import Final
import sys

_T = TypeVar("_T")

class FDCapture(Generic[AnyStr]):
    def __init__(self, targetfd: int, tmpfile: Optional[IO[AnyStr]] = ..., now: bool = ..., patchsys: bool = ...) -> None: ...
    def start(self) -> None: ...
    def done(self) -> IO[AnyStr]: ...
    def writeorg(self, data: AnyStr) -> None: ...

class StdCaptureFD:
    def __init__(
        self,
        out: Union[bool, IO[str]] = ...,
        err: Union[bool, IO[str]] = ...,
        mixed: bool = ...,
        in_: bool = ...,
        patchsys: bool = ...,
        now: bool = ...,
    ) -> None: ...
    @classmethod
    def call(cls, func: Callable[..., _T], *args: Any, **kwargs: Any) -> Tuple[_T, str, str]: ...
    def reset(self) -> Tuple[str, str]: ...
    def suspend(self) -> Tuple[str, str]: ...
    def startall(self) -> None: ...
    def resume(self) -> None: ...
    def done(self, save: bool = ...) -> Tuple[IO[str], IO[str]]: ...
    def readouterr(self) -> Tuple[str, str]: ...

class StdCapture:
    def __init__(
        self,
        out: Union[bool, IO[str]] = ...,
        err: Union[bool, IO[str]] = ...,
        in_: bool = ...,
        mixed: bool = ...,
        now: bool = ...,
    ) -> None: ...
    @classmethod
    def call(cls, func: Callable[..., _T], *args: Any, **kwargs: Any) -> Tuple[_T, str, str]: ...
    def reset(self) -> Tuple[str, str]: ...
    def suspend(self) -> Tuple[str, str]: ...
    def startall(self) -> None: ...
    def resume(self) -> None: ...
    def done(self, save: bool = ...) -> Tuple[IO[str], IO[str]]: ...
    def readouterr(self) -> Tuple[IO[str], IO[str]]: ...

# XXX: The type here is not exactly right. If f is IO[bytes] and
# encoding is not None, returns some weird hybrid, not exactly IO[bytes].
def dupfile(
    f: IO[AnyStr],
    mode: Optional[str] = ...,
    buffering: int = ...,
    raising: bool = ...,
    encoding: Optional[str] = ...,
) -> IO[AnyStr]: ...
def get_terminal_width() -> int: ...
def ansi_print(
    text: Union[str, Text],
    esc: Union[Union[str, Text], Tuple[Union[str, Text], ...]],
    file: Optional[IO[Any]] = ...,
    newline: bool = ...,
    flush: bool = ...,
) -> None: ...
def saferepr(obj, maxsize: int = ...) -> str: ...

class TerminalWriter:
    stringio: TextIO
    encoding: Final[str]
    hasmarkup: bool
    def __init__(self, file: Optional[IO[str]] = ..., stringio: bool = ..., encoding: Optional[str] = ...) -> None: ...
    @property
    def fullwidth(self) -> int: ...
    @fullwidth.setter
    def fullwidth(self, value: int) -> None: ...
    @property
    def chars_on_current_line(self) -> int: ...
    @property
    def width_of_current_line(self) -> int: ...
    def markup(
        self,
        text: str,
        *,
        black: int = ..., red: int = ..., green: int = ..., yellow: int = ..., blue: int = ..., purple: int = ...,
        cyan: int = ..., white: int = ..., Black: int = ..., Red: int = ..., Green: int = ..., Yellow: int = ...,
        Blue: int = ..., Purple: int = ..., Cyan: int = ..., White: int = ..., bold: int = ..., light: int = ...,
        blink: int = ..., invert: int = ...,
    ) -> str: ...
    def sep(
        self,
        sepchar: str,
        title: Optional[str] = ...,
        fullwidth: Optional[int] = ...,
        *,
        black: int = ..., red: int = ..., green: int = ..., yellow: int = ..., blue: int = ..., purple: int = ...,
        cyan: int = ..., white: int = ..., Black: int = ..., Red: int = ..., Green: int = ..., Yellow: int = ...,
        Blue: int = ..., Purple: int = ..., Cyan: int = ..., White: int = ..., bold: int = ..., light: int = ...,
        blink: int = ..., invert: int = ...,
    ) -> None: ...
    def write(
        self,
        msg: str,
        *,
        black: int = ..., red: int = ..., green: int = ..., yellow: int = ..., blue: int = ..., purple: int = ...,
        cyan: int = ..., white: int = ..., Black: int = ..., Red: int = ..., Green: int = ..., Yellow: int = ...,
        Blue: int = ..., Purple: int = ..., Cyan: int = ..., White: int = ..., bold: int = ..., light: int = ...,
        blink: int = ..., invert: int = ...,
    ) -> None: ...
    def line(
        self,
        s: str = ...,
        *,
        black: int = ..., red: int = ..., green: int = ..., yellow: int = ..., blue: int = ..., purple: int = ...,
        cyan: int = ..., white: int = ..., Black: int = ..., Red: int = ..., Green: int = ..., Yellow: int = ...,
        Blue: int = ..., Purple: int = ..., Cyan: int = ..., White: int = ..., bold: int = ..., light: int = ...,
        blink: int = ..., invert: int = ...,
    ) -> None: ...
    def reline(
        self,
        line: str,
        *,
        black: int = ..., red: int = ..., green: int = ..., yellow: int = ..., blue: int = ..., purple: int = ...,
        cyan: int = ..., white: int = ..., Black: int = ..., Red: int = ..., Green: int = ..., Yellow: int = ...,
        Blue: int = ..., Purple: int = ..., Cyan: int = ..., White: int = ..., bold: int = ..., light: int = ...,
        blink: int = ..., invert: int = ...,
    ) -> None: ...
