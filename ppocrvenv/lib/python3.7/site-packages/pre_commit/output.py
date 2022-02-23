import contextlib
import sys
from typing import Any
from typing import IO
from typing import Optional


def write(s: str, stream: IO[bytes] = sys.stdout.buffer) -> None:
    stream.write(s.encode())
    stream.flush()


def write_line_b(
        s: Optional[bytes] = None,
        stream: IO[bytes] = sys.stdout.buffer,
        logfile_name: Optional[str] = None,
) -> None:
    with contextlib.ExitStack() as exit_stack:
        output_streams = [stream]
        if logfile_name:
            stream = exit_stack.enter_context(open(logfile_name, 'ab'))
            output_streams.append(stream)

        for output_stream in output_streams:
            if s is not None:
                output_stream.write(s)
            output_stream.write(b'\n')
            output_stream.flush()


def write_line(s: Optional[str] = None, **kwargs: Any) -> None:
    write_line_b(s.encode() if s is not None else s, **kwargs)
