import re
from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple

from .._internal import _to_bytes
from .._internal import _to_str
from ..datastructures import Headers
from ..exceptions import RequestEntityTooLarge
from ..http import parse_options_header


class Event:
    pass


@dataclass(frozen=True)
class Preamble(Event):
    data: bytes


@dataclass(frozen=True)
class Field(Event):
    name: str
    headers: Headers


@dataclass(frozen=True)
class File(Event):
    name: str
    filename: str
    headers: Headers


@dataclass(frozen=True)
class Data(Event):
    data: bytes
    more_data: bool


@dataclass(frozen=True)
class Epilogue(Event):
    data: bytes


class NeedData(Event):
    pass


NEED_DATA = NeedData()


class State(Enum):
    PREAMBLE = auto()
    PART = auto()
    DATA = auto()
    EPILOGUE = auto()
    COMPLETE = auto()


# Multipart line breaks MUST be CRLF (\r\n) by RFC-7578, except that
# many implementations break this and either use CR or LF alone.
LINE_BREAK = b"(?:\r\n|\n|\r)"
BLANK_LINE_RE = re.compile(b"(?:\r\n\r\n|\r\r|\n\n)", re.MULTILINE)
LINE_BREAK_RE = re.compile(LINE_BREAK, re.MULTILINE)
# Header values can be continued via a space or tab after the linebreak, as
# per RFC2231
HEADER_CONTINUATION_RE = re.compile(b"%s[ \t]" % LINE_BREAK, re.MULTILINE)


class MultipartDecoder:
    """Decodes a multipart message as bytes into Python events.

    The part data is returned as available to allow the caller to save
    the data from memory to disk, if desired.
    """

    def __init__(
        self,
        boundary: bytes,
        max_form_memory_size: Optional[int] = None,
    ) -> None:
        self.buffer = bytearray()
        self.complete = False
        self.max_form_memory_size = max_form_memory_size
        self.state = State.PREAMBLE
        self.boundary = boundary

        # Note in the below \h i.e. horizontal whitespace is used
        # as [^\S\n\r] as \h isn't supported in python.

        # The preamble must end with a boundary where the boundary is
        # prefixed by a line break, RFC2046. Except that many
        # implementations including Werkzeug's tests omit the line
        # break prefix. In addition the first boundary could be the
        # epilogue boundary (for empty form-data) hence the matching
        # group to understand if it is an epilogue boundary.
        self.preamble_re = re.compile(
            br"%s?--%s(--[^\S\n\r]*%s?|[^\S\n\r]*%s)"
            % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK),
            re.MULTILINE,
        )
        # A boundary must include a line break prefix and suffix, and
        # may include trailing whitespace. In addition the boundary
        # could be the epilogue boundary hence the matching group to
        # understand if it is an epilogue boundary.
        self.boundary_re = re.compile(
            br"%s--%s(--[^\S\n\r]*%s?|[^\S\n\r]*%s)"
            % (LINE_BREAK, re.escape(boundary), LINE_BREAK, LINE_BREAK),
            re.MULTILINE,
        )

    def last_newline(self) -> int:
        try:
            last_nl = self.buffer.rindex(b"\n")
        except ValueError:
            last_nl = len(self.buffer)
        try:
            last_cr = self.buffer.rindex(b"\r")
        except ValueError:
            last_cr = len(self.buffer)

        return min(last_nl, last_cr)

    def receive_data(self, data: Optional[bytes]) -> None:
        if data is None:
            self.complete = True
        elif (
            self.max_form_memory_size is not None
            and len(self.buffer) + len(data) > self.max_form_memory_size
        ):
            raise RequestEntityTooLarge()
        else:
            self.buffer.extend(data)

    def next_event(self) -> Event:
        event: Event = NEED_DATA

        if self.state == State.PREAMBLE:
            match = self.preamble_re.search(self.buffer)
            if match is not None:
                if match.group(1).startswith(b"--"):
                    self.state = State.EPILOGUE
                else:
                    self.state = State.PART
                data = bytes(self.buffer[: match.start()])
                del self.buffer[: match.end()]
                event = Preamble(data=data)

        elif self.state == State.PART:
            match = BLANK_LINE_RE.search(self.buffer)
            if match is not None:
                headers = self._parse_headers(self.buffer[: match.start()])
                del self.buffer[: match.end()]

                if "content-disposition" not in headers:
                    raise ValueError("Missing Content-Disposition header")

                disposition, extra = parse_options_header(
                    headers["content-disposition"]
                )
                name = cast(str, extra.get("name"))
                filename = extra.get("filename")
                if filename is not None:
                    event = File(
                        filename=filename,
                        headers=headers,
                        name=name,
                    )
                else:
                    event = Field(
                        headers=headers,
                        name=name,
                    )
                self.state = State.DATA

        elif self.state == State.DATA:
            if self.buffer.find(b"--" + self.boundary) == -1:
                # No complete boundary in the buffer, but there may be
                # a partial boundary at the end. As the boundary
                # starts with either a nl or cr find the earliest and
                # return up to that as data.
                data_length = del_index = self.last_newline()
                more_data = True
            else:
                match = self.boundary_re.search(self.buffer)
                if match is not None:
                    if match.group(1).startswith(b"--"):
                        self.state = State.EPILOGUE
                    else:
                        self.state = State.PART
                    data_length = match.start()
                    del_index = match.end()
                else:
                    data_length = del_index = self.last_newline()
                more_data = match is None

            data = bytes(self.buffer[:data_length])
            del self.buffer[:del_index]
            if data or not more_data:
                event = Data(data=data, more_data=more_data)

        elif self.state == State.EPILOGUE and self.complete:
            event = Epilogue(data=bytes(self.buffer))
            del self.buffer[:]
            self.state = State.COMPLETE

        if self.complete and isinstance(event, NeedData):
            raise ValueError(f"Invalid form-data cannot parse beyond {self.state}")

        return event

    def _parse_headers(self, data: bytes) -> Headers:
        headers: List[Tuple[str, str]] = []
        # Merge the continued headers into one line
        data = HEADER_CONTINUATION_RE.sub(b" ", data)
        # Now there is one header per line
        for line in data.splitlines():
            if line.strip() != b"":
                name, value = _to_str(line).strip().split(":", 1)
                headers.append((name.strip(), value.strip()))
        return Headers(headers)


class MultipartEncoder:
    def __init__(self, boundary: bytes) -> None:
        self.boundary = boundary
        self.state = State.PREAMBLE

    def send_event(self, event: Event) -> bytes:
        if isinstance(event, Preamble) and self.state == State.PREAMBLE:
            self.state = State.PART
            return event.data
        elif isinstance(event, (Field, File)) and self.state in {
            State.PREAMBLE,
            State.PART,
            State.DATA,
        }:
            self.state = State.DATA
            data = b"\r\n--" + self.boundary + b"\r\n"
            data += b'Content-Disposition: form-data; name="%s"' % _to_bytes(event.name)
            if isinstance(event, File):
                data += b'; filename="%s"' % _to_bytes(event.filename)
            data += b"\r\n"
            for name, value in cast(Field, event).headers:
                if name.lower() != "content-disposition":
                    data += _to_bytes(f"{name}: {value}\r\n")
            data += b"\r\n"
            return data
        elif isinstance(event, Data) and self.state == State.DATA:
            return event.data
        elif isinstance(event, Epilogue):
            self.state = State.COMPLETE
            return b"\r\n--" + self.boundary + b"--\r\n" + event.data
        else:
            raise ValueError(f"Cannot generate {event} in state: {self.state}")
