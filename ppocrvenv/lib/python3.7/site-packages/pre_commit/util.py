import contextlib
import errno
import functools
import os.path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import IO
from typing import Optional
from typing import Tuple
from typing import Type

import yaml

from pre_commit import parse_shebang

if sys.version_info >= (3, 7):  # pragma: >=3.7 cover
    from importlib.resources import open_binary
    from importlib.resources import read_text
else:  # pragma: <3.7 cover
    from importlib_resources import open_binary
    from importlib_resources import read_text

Loader = getattr(yaml, 'CSafeLoader', yaml.SafeLoader)
yaml_load = functools.partial(yaml.load, Loader=Loader)
Dumper = getattr(yaml, 'CSafeDumper', yaml.SafeDumper)


def yaml_dump(o: Any, **kwargs: Any) -> str:
    # when python/mypy#1484 is solved, this can be `functools.partial`
    return yaml.dump(
        o, Dumper=Dumper, default_flow_style=False, indent=4, sort_keys=False,
        **kwargs,
    )


def force_bytes(exc: Any) -> bytes:
    with contextlib.suppress(TypeError):
        return bytes(exc)
    with contextlib.suppress(Exception):
        return str(exc).encode()
    return f'<unprintable {type(exc).__name__} object>'.encode()


@contextlib.contextmanager
def clean_path_on_failure(path: str) -> Generator[None, None, None]:
    """Cleans up the directory on an exceptional failure."""
    try:
        yield
    except BaseException:
        if os.path.exists(path):
            rmtree(path)
        raise


@contextlib.contextmanager
def tmpdir() -> Generator[str, None, None]:
    """Contextmanager to create a temporary directory.  It will be cleaned up
    afterwards.
    """
    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
    finally:
        rmtree(tempdir)


def resource_bytesio(filename: str) -> IO[bytes]:
    return open_binary('pre_commit.resources', filename)


def resource_text(filename: str) -> str:
    return read_text('pre_commit.resources', filename)


def make_executable(filename: str) -> None:
    original_mode = os.stat(filename).st_mode
    new_mode = original_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    os.chmod(filename, new_mode)


class CalledProcessError(RuntimeError):
    def __init__(
            self,
            returncode: int,
            cmd: Tuple[str, ...],
            expected_returncode: int,
            stdout: bytes,
            stderr: Optional[bytes],
    ) -> None:
        super().__init__(returncode, cmd, expected_returncode, stdout, stderr)
        self.returncode = returncode
        self.cmd = cmd
        self.expected_returncode = expected_returncode
        self.stdout = stdout
        self.stderr = stderr

    def __bytes__(self) -> bytes:
        def _indent_or_none(part: Optional[bytes]) -> bytes:
            if part:
                return b'\n    ' + part.replace(b'\n', b'\n    ')
            else:
                return b' (none)'

        return b''.join((
            f'command: {self.cmd!r}\n'.encode(),
            f'return code: {self.returncode}\n'.encode(),
            f'expected return code: {self.expected_returncode}\n'.encode(),
            b'stdout:', _indent_or_none(self.stdout), b'\n',
            b'stderr:', _indent_or_none(self.stderr),
        ))

    def __str__(self) -> str:
        return self.__bytes__().decode()


def _setdefault_kwargs(kwargs: Dict[str, Any]) -> None:
    for arg in ('stdin', 'stdout', 'stderr'):
        kwargs.setdefault(arg, subprocess.PIPE)


def _oserror_to_output(e: OSError) -> Tuple[int, bytes, None]:
    return 1, force_bytes(e).rstrip(b'\n') + b'\n', None


def cmd_output_b(
        *cmd: str,
        retcode: Optional[int] = 0,
        **kwargs: Any,
) -> Tuple[int, bytes, Optional[bytes]]:
    _setdefault_kwargs(kwargs)

    try:
        cmd = parse_shebang.normalize_cmd(cmd)
    except parse_shebang.ExecutableNotFoundError as e:
        returncode, stdout_b, stderr_b = e.to_output()
    else:
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except OSError as e:
            returncode, stdout_b, stderr_b = _oserror_to_output(e)
        else:
            stdout_b, stderr_b = proc.communicate()
            returncode = proc.returncode

    if retcode is not None and retcode != returncode:
        raise CalledProcessError(returncode, cmd, retcode, stdout_b, stderr_b)

    return returncode, stdout_b, stderr_b


def cmd_output(*cmd: str, **kwargs: Any) -> Tuple[int, str, Optional[str]]:
    returncode, stdout_b, stderr_b = cmd_output_b(*cmd, **kwargs)
    stdout = stdout_b.decode() if stdout_b is not None else None
    stderr = stderr_b.decode() if stderr_b is not None else None
    return returncode, stdout, stderr


if os.name != 'nt':  # pragma: win32 no cover
    from os import openpty
    import termios

    class Pty:
        def __init__(self) -> None:
            self.r: Optional[int] = None
            self.w: Optional[int] = None

        def __enter__(self) -> 'Pty':
            self.r, self.w = openpty()

            # tty flags normally change \n to \r\n
            attrs = termios.tcgetattr(self.r)
            assert isinstance(attrs[1], int)
            attrs[1] &= ~(termios.ONLCR | termios.OPOST)
            termios.tcsetattr(self.r, termios.TCSANOW, attrs)

            return self

        def close_w(self) -> None:
            if self.w is not None:
                os.close(self.w)
                self.w = None

        def close_r(self) -> None:
            assert self.r is not None
            os.close(self.r)
            self.r = None

        def __exit__(
                self,
                exc_type: Optional[Type[BaseException]],
                exc_value: Optional[BaseException],
                traceback: Optional[TracebackType],
        ) -> None:
            self.close_w()
            self.close_r()

    def cmd_output_p(
            *cmd: str,
            retcode: Optional[int] = 0,
            **kwargs: Any,
    ) -> Tuple[int, bytes, Optional[bytes]]:
        assert retcode is None
        assert kwargs['stderr'] == subprocess.STDOUT, kwargs['stderr']
        _setdefault_kwargs(kwargs)

        try:
            cmd = parse_shebang.normalize_cmd(cmd)
        except parse_shebang.ExecutableNotFoundError as e:
            return e.to_output()

        with open(os.devnull) as devnull, Pty() as pty:
            assert pty.r is not None
            kwargs.update({'stdin': devnull, 'stdout': pty.w, 'stderr': pty.w})
            try:
                proc = subprocess.Popen(cmd, **kwargs)
            except OSError as e:
                return _oserror_to_output(e)

            pty.close_w()

            buf = b''
            while True:
                try:
                    bts = os.read(pty.r, 4096)
                except OSError as e:
                    if e.errno == errno.EIO:
                        bts = b''
                    else:
                        raise
                else:
                    buf += bts
                if not bts:
                    break

        return proc.wait(), buf, None
else:  # pragma: no cover
    cmd_output_p = cmd_output_b


def rmtree(path: str) -> None:
    """On windows, rmtree fails for readonly dirs."""
    def handle_remove_readonly(
            func: Callable[..., Any],
            path: str,
            exc: Tuple[Type[OSError], OSError, TracebackType],
    ) -> None:
        excvalue = exc[1]
        if (
                func in (os.rmdir, os.remove, os.unlink) and
                excvalue.errno in {errno.EACCES, errno.EPERM}
        ):
            for p in (path, os.path.dirname(path)):
                os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
            func(path)
        else:
            raise
    shutil.rmtree(path, ignore_errors=False, onerror=handle_remove_readonly)


def parse_version(s: str) -> Tuple[int, ...]:
    """poor man's version comparison"""
    return tuple(int(p) for p in s.split('.'))


def win_exe(s: str) -> str:
    return s if sys.platform != 'win32' else f'{s}.exe'
