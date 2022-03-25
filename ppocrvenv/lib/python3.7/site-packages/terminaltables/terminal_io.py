"""Get info about the current terminal window/screen buffer."""

import ctypes
import struct
import sys

DEFAULT_HEIGHT = 24
DEFAULT_WIDTH = 79
INVALID_HANDLE_VALUE = -1
IS_WINDOWS = sys.platform == 'win32'
STD_ERROR_HANDLE = -12
STD_OUTPUT_HANDLE = -11


def get_console_info(kernel32, handle):
    """Get information about this current console window (Windows only).

    https://github.com/Robpol86/colorclass/blob/ab42da59/colorclass/windows.py#L111

    :raise OSError: When handle is invalid or GetConsoleScreenBufferInfo API call fails.

    :param ctypes.windll.kernel32 kernel32: Loaded kernel32 instance.
    :param int handle: stderr or stdout handle.

    :return: Width (number of characters) and height (number of lines) of the terminal.
    :rtype: tuple
    """
    if handle == INVALID_HANDLE_VALUE:
        raise OSError('Invalid handle.')

    # Query Win32 API.
    lpcsbi = ctypes.create_string_buffer(22)  # Populated by GetConsoleScreenBufferInfo.
    if not kernel32.GetConsoleScreenBufferInfo(handle, lpcsbi):
        raise ctypes.WinError()  # Subclass of OSError.

    # Parse data.
    left, top, right, bottom = struct.unpack('hhhhHhhhhhh', lpcsbi.raw)[5:-2]
    width, height = right - left, bottom - top
    return width, height


def terminal_size(kernel32=None):
    """Get the width and height of the terminal.

    http://code.activestate.com/recipes/440694-determine-size-of-console-window-on-windows/
    http://stackoverflow.com/questions/17993814/why-the-irrelevant-code-made-a-difference

    :param kernel32: Optional mock kernel32 object. For testing.

    :return: Width (number of characters) and height (number of lines) of the terminal.
    :rtype: tuple
    """
    if IS_WINDOWS:
        kernel32 = kernel32 or ctypes.windll.kernel32
        try:
            return get_console_info(kernel32, kernel32.GetStdHandle(STD_ERROR_HANDLE))
        except OSError:
            try:
                return get_console_info(kernel32, kernel32.GetStdHandle(STD_OUTPUT_HANDLE))
            except OSError:
                return DEFAULT_WIDTH, DEFAULT_HEIGHT

    try:
        device = __import__('fcntl').ioctl(0, __import__('termios').TIOCGWINSZ, '\0\0\0\0\0\0\0\0')
    except IOError:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT
    height, width = struct.unpack('hhhh', device)[:2]
    return width, height


def set_terminal_title(title, kernel32=None):
    """Set the terminal title.

    :param title: The title to set (string, unicode, bytes accepted).
    :param kernel32: Optional mock kernel32 object. For testing.

    :return: If title changed successfully (Windows only, always True on Linux/OSX).
    :rtype: bool
    """
    try:
        title_bytes = title.encode('utf-8')
    except AttributeError:
        title_bytes = title

    if IS_WINDOWS:
        kernel32 = kernel32 or ctypes.windll.kernel32
        try:
            is_ascii = all(ord(c) < 128 for c in title)  # str/unicode.
        except TypeError:
            is_ascii = all(c < 128 for c in title)  # bytes.
        if is_ascii:
            return kernel32.SetConsoleTitleA(title_bytes) != 0
        else:
            return kernel32.SetConsoleTitleW(title) != 0

    # Linux/OSX.
    sys.stdout.write(b'\033]0;' + title_bytes + b'\007')
    return True
