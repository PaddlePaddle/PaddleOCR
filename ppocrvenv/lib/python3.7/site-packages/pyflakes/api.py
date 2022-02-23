"""
API for the command-line I{pyflakes} tool.
"""
from __future__ import with_statement

import ast
import os
import platform
import re
import sys

from pyflakes import checker, __version__
from pyflakes import reporter as modReporter

__all__ = ['check', 'checkPath', 'checkRecursive', 'iterSourceCode', 'main']

PYTHON_SHEBANG_REGEX = re.compile(br'^#!.*\bpython([23](\.\d+)?|w)?[dmu]?\s')


def check(codeString, filename, reporter=None):
    """
    Check the Python source given by C{codeString} for flakes.

    @param codeString: The Python source to check.
    @type codeString: C{str}

    @param filename: The name of the file the source came from, used to report
        errors.
    @type filename: C{str}

    @param reporter: A L{Reporter} instance, where errors and warnings will be
        reported.

    @return: The number of warnings emitted.
    @rtype: C{int}
    """
    if reporter is None:
        reporter = modReporter._makeDefaultReporter()
    # First, compile into an AST and handle syntax errors.
    try:
        tree = ast.parse(codeString, filename=filename)
    except SyntaxError:
        value = sys.exc_info()[1]
        msg = value.args[0]

        (lineno, offset, text) = value.lineno, value.offset, value.text

        if checker.PYPY:
            if text is None:
                lines = codeString.splitlines()
                if len(lines) >= lineno:
                    text = lines[lineno - 1]
                    if sys.version_info >= (3, ) and isinstance(text, bytes):
                        try:
                            text = text.decode('ascii')
                        except UnicodeDecodeError:
                            text = None
            offset -= 1

        # If there's an encoding problem with the file, the text is None.
        if text is None:
            # Avoid using msg, since for the only known case, it contains a
            # bogus message that claims the encoding the file declared was
            # unknown.
            reporter.unexpectedError(filename, 'problem decoding source')
        else:
            reporter.syntaxError(filename, msg, lineno, offset, text)
        return 1
    except Exception:
        reporter.unexpectedError(filename, 'problem decoding source')
        return 1
    # Okay, it's syntactically valid.  Now check it.
    file_tokens = checker.make_tokens(codeString)
    w = checker.Checker(tree, file_tokens=file_tokens, filename=filename)
    w.messages.sort(key=lambda m: m.lineno)
    for warning in w.messages:
        reporter.flake(warning)
    return len(w.messages)


def checkPath(filename, reporter=None):
    """
    Check the given path, printing out any warnings detected.

    @param reporter: A L{Reporter} instance, where errors and warnings will be
        reported.

    @return: the number of warnings printed
    """
    if reporter is None:
        reporter = modReporter._makeDefaultReporter()
    try:
        with open(filename, 'rb') as f:
            codestr = f.read()
    except IOError:
        msg = sys.exc_info()[1]
        reporter.unexpectedError(filename, msg.args[1])
        return 1
    return check(codestr, filename, reporter)


def isPythonFile(filename):
    """Return True if filename points to a Python file."""
    if filename.endswith('.py'):
        return True

    # Avoid obvious Emacs backup files
    if filename.endswith("~"):
        return False

    max_bytes = 128

    try:
        with open(filename, 'rb') as f:
            text = f.read(max_bytes)
            if not text:
                return False
    except IOError:
        return False

    return PYTHON_SHEBANG_REGEX.match(text)


def iterSourceCode(paths):
    """
    Iterate over all Python source files in C{paths}.

    @param paths: A list of paths.  Directories will be recursed into and
        any .py files found will be yielded.  Any non-directories will be
        yielded as-is.
    """
    for path in paths:
        if os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    full_path = os.path.join(dirpath, filename)
                    if isPythonFile(full_path):
                        yield full_path
        else:
            yield path


def checkRecursive(paths, reporter):
    """
    Recursively check all source files in C{paths}.

    @param paths: A list of paths to Python source files and directories
        containing Python source files.
    @param reporter: A L{Reporter} where all of the warnings and errors
        will be reported to.
    @return: The number of warnings found.
    """
    warnings = 0
    for sourcePath in iterSourceCode(paths):
        warnings += checkPath(sourcePath, reporter)
    return warnings


def _exitOnSignal(sigName, message):
    """Handles a signal with sys.exit.

    Some of these signals (SIGPIPE, for example) don't exist or are invalid on
    Windows. So, ignore errors that might arise.
    """
    import signal

    try:
        sigNumber = getattr(signal, sigName)
    except AttributeError:
        # the signal constants defined in the signal module are defined by
        # whether the C library supports them or not. So, SIGPIPE might not
        # even be defined.
        return

    def handler(sig, f):
        sys.exit(message)

    try:
        signal.signal(sigNumber, handler)
    except ValueError:
        # It's also possible the signal is defined, but then it's invalid. In
        # this case, signal.signal raises ValueError.
        pass


def _get_version():
    """
    Retrieve and format package version along with python version & OS used
    """
    return ('%s Python %s on %s' %
            (__version__, platform.python_version(), platform.system()))


def main(prog=None, args=None):
    """Entry point for the script "pyflakes"."""
    import argparse

    # Handle "Keyboard Interrupt" and "Broken pipe" gracefully
    _exitOnSignal('SIGINT', '... stopped')
    _exitOnSignal('SIGPIPE', 1)

    parser = argparse.ArgumentParser(prog=prog,
                                     description='Check Python source files for errors')
    parser.add_argument('-V', '--version', action='version', version=_get_version())
    parser.add_argument('path', nargs='*',
                        help='Path(s) of Python file(s) to check. STDIN if not given.')
    args = parser.parse_args(args=args).path
    reporter = modReporter._makeDefaultReporter()
    if args:
        warnings = checkRecursive(args, reporter)
    else:
        warnings = check(sys.stdin.read(), '<stdin>', reporter)
    raise SystemExit(warnings > 0)
