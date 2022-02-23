"""Utility methods for flake8."""
import collections
import fnmatch as _fnmatch
import functools
import inspect
import io
import logging
import os
import platform
import re
import sys
import textwrap
import tokenize
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

from flake8 import exceptions

if TYPE_CHECKING:
    from flake8.plugins.manager import Plugin

DIFF_HUNK_REGEXP = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@.*$")
COMMA_SEPARATED_LIST_RE = re.compile(r"[,\s]")
LOCAL_PLUGIN_LIST_RE = re.compile(r"[,\t\n\r\f\v]")


def parse_comma_separated_list(
    value: str, regexp: Pattern[str] = COMMA_SEPARATED_LIST_RE
) -> List[str]:
    """Parse a comma-separated list.

    :param value:
        String to be parsed and normalized.
    :param regexp:
        Compiled regular expression used to split the value when it is a
        string.
    :type regexp:
        _sre.SRE_Pattern
    :returns:
        List of values with whitespace stripped.
    :rtype:
        list
    """
    assert isinstance(value, str), value

    separated = regexp.split(value)
    item_gen = (item.strip() for item in separated)
    return [item for item in item_gen if item]


_Token = collections.namedtuple("_Token", ("tp", "src"))
_CODE, _FILE, _COLON, _COMMA, _WS = "code", "file", "colon", "comma", "ws"
_EOF = "eof"
_FILE_LIST_TOKEN_TYPES = [
    (re.compile(r"[A-Z]+[0-9]*(?=$|\s|,)"), _CODE),
    (re.compile(r"[^\s:,]+"), _FILE),
    (re.compile(r"\s*:\s*"), _COLON),
    (re.compile(r"\s*,\s*"), _COMMA),
    (re.compile(r"\s+"), _WS),
]


def _tokenize_files_to_codes_mapping(value: str) -> List[_Token]:
    tokens = []
    i = 0
    while i < len(value):
        for token_re, token_name in _FILE_LIST_TOKEN_TYPES:
            match = token_re.match(value, i)
            if match:
                tokens.append(_Token(token_name, match.group().strip()))
                i = match.end()
                break
        else:
            raise AssertionError("unreachable", value, i)
    tokens.append(_Token(_EOF, ""))

    return tokens


def parse_files_to_codes_mapping(  # noqa: C901
    value_: Union[Sequence[str], str]
) -> List[Tuple[str, List[str]]]:
    """Parse a files-to-codes mapping.

    A files-to-codes mapping a sequence of values specified as
    `filenames list:codes list ...`.  Each of the lists may be separated by
    either comma or whitespace tokens.

    :param value: String to be parsed and normalized.
    :type value: str
    """
    if not isinstance(value_, str):
        value = "\n".join(value_)
    else:
        value = value_

    ret: List[Tuple[str, List[str]]] = []
    if not value.strip():
        return ret

    class State:
        seen_sep = True
        seen_colon = False
        filenames: List[str] = []
        codes: List[str] = []

    def _reset() -> None:
        if State.codes:
            for filename in State.filenames:
                ret.append((filename, State.codes))
        State.seen_sep = True
        State.seen_colon = False
        State.filenames = []
        State.codes = []

    def _unexpected_token() -> exceptions.ExecutionError:
        return exceptions.ExecutionError(
            f"Expected `per-file-ignores` to be a mapping from file exclude "
            f"patterns to ignore codes.\n\n"
            f"Configured `per-file-ignores` setting:\n\n"
            f"{textwrap.indent(value.strip(), '    ')}"
        )

    for token in _tokenize_files_to_codes_mapping(value):
        # legal in any state: separator sets the sep bit
        if token.tp in {_COMMA, _WS}:
            State.seen_sep = True
        # looking for filenames
        elif not State.seen_colon:
            if token.tp == _COLON:
                State.seen_colon = True
                State.seen_sep = True
            elif State.seen_sep and token.tp == _FILE:
                State.filenames.append(token.src)
                State.seen_sep = False
            else:
                raise _unexpected_token()
        # looking for codes
        else:
            if token.tp == _EOF:
                _reset()
            elif State.seen_sep and token.tp == _CODE:
                State.codes.append(token.src)
                State.seen_sep = False
            elif State.seen_sep and token.tp == _FILE:
                _reset()
                State.filenames.append(token.src)
                State.seen_sep = False
            else:
                raise _unexpected_token()

    return ret


def normalize_paths(
    paths: Sequence[str], parent: str = os.curdir
) -> List[str]:
    """Normalize a list of paths relative to a parent directory.

    :returns:
        The normalized paths.
    :rtype:
        [str]
    """
    assert isinstance(paths, list), paths
    return [normalize_path(p, parent) for p in paths]


def normalize_path(path: str, parent: str = os.curdir) -> str:
    """Normalize a single-path.

    :returns:
        The normalized path.
    :rtype:
        str
    """
    # NOTE(sigmavirus24): Using os.path.sep and os.path.altsep allow for
    # Windows compatibility with both Windows-style paths (c:\\foo\bar) and
    # Unix style paths (/foo/bar).
    separator = os.path.sep
    # NOTE(sigmavirus24): os.path.altsep may be None
    alternate_separator = os.path.altsep or ""
    if separator in path or (
        alternate_separator and alternate_separator in path
    ):
        path = os.path.abspath(os.path.join(parent, path))
    return path.rstrip(separator + alternate_separator)


@functools.lru_cache(maxsize=1)
def stdin_get_value() -> str:
    """Get and cache it so plugins can use it."""
    stdin_value = sys.stdin.buffer.read()
    fd = io.BytesIO(stdin_value)
    try:
        coding, _ = tokenize.detect_encoding(fd.readline)
        fd.seek(0)
        return io.TextIOWrapper(fd, coding).read()
    except (LookupError, SyntaxError, UnicodeError):
        return stdin_value.decode("utf-8")


def stdin_get_lines() -> List[str]:
    """Return lines of stdin split according to file splitting."""
    return list(io.StringIO(stdin_get_value()))


def parse_unified_diff(diff: Optional[str] = None) -> Dict[str, Set[int]]:
    """Parse the unified diff passed on stdin.

    :returns:
        dictionary mapping file names to sets of line numbers
    :rtype:
        dict
    """
    # Allow us to not have to patch out stdin_get_value
    if diff is None:
        diff = stdin_get_value()

    number_of_rows = None
    current_path = None
    parsed_paths: Dict[str, Set[int]] = collections.defaultdict(set)
    for line in diff.splitlines():
        if number_of_rows:
            if not line or line[0] != "-":
                number_of_rows -= 1
            # We're in the part of the diff that has lines starting with +, -,
            # and ' ' to show context and the changes made. We skip these
            # because the information we care about is the filename and the
            # range within it.
            # When number_of_rows reaches 0, we will once again start
            # searching for filenames and ranges.
            continue

        # NOTE(sigmavirus24): Diffs that we support look roughly like:
        #    diff a/file.py b/file.py
        #    ...
        #    --- a/file.py
        #    +++ b/file.py
        # Below we're looking for that last line. Every diff tool that
        # gives us this output may have additional information after
        # ``b/file.py`` which it will separate with a \t, e.g.,
        #    +++ b/file.py\t100644
        # Which is an example that has the new file permissions/mode.
        # In this case we only care about the file name.
        if line[:3] == "+++":
            current_path = line[4:].split("\t", 1)[0]
            # NOTE(sigmavirus24): This check is for diff output from git.
            if current_path[:2] == "b/":
                current_path = current_path[2:]
            # We don't need to do anything else. We have set up our local
            # ``current_path`` variable. We can skip the rest of this loop.
            # The next line we will see will give us the hung information
            # which is in the next section of logic.
            continue

        hunk_match = DIFF_HUNK_REGEXP.match(line)
        # NOTE(sigmavirus24): pep8/pycodestyle check for:
        #    line[:3] == '@@ '
        # But the DIFF_HUNK_REGEXP enforces that the line start with that
        # So we can more simply check for a match instead of slicing and
        # comparing.
        if hunk_match:
            (row, number_of_rows) = (
                1 if not group else int(group) for group in hunk_match.groups()
            )
            assert current_path is not None
            parsed_paths[current_path].update(range(row, row + number_of_rows))

    # We have now parsed our diff into a dictionary that looks like:
    #    {'file.py': set(range(10, 16), range(18, 20)), ...}
    return parsed_paths


def is_windows() -> bool:
    """Determine if we're running on Windows.

    :returns:
        True if running on Windows, otherwise False
    :rtype:
        bool
    """
    return os.name == "nt"


def is_using_stdin(paths: List[str]) -> bool:
    """Determine if we're going to read from stdin.

    :param list paths:
        The paths that we're going to check.
    :returns:
        True if stdin (-) is in the path, otherwise False
    :rtype:
        bool
    """
    return "-" in paths


def _default_predicate(*args: str) -> bool:
    return False


def filenames_from(
    arg: str, predicate: Optional[Callable[[str], bool]] = None
) -> Generator[str, None, None]:
    """Generate filenames from an argument.

    :param str arg:
        Parameter from the command-line.
    :param callable predicate:
        Predicate to use to filter out filenames. If the predicate
        returns ``True`` we will exclude the filename, otherwise we
        will yield it. By default, we include every filename
        generated.
    :returns:
        Generator of paths
    """
    if predicate is None:
        predicate = _default_predicate

    if predicate(arg):
        return

    if os.path.isdir(arg):
        for root, sub_directories, files in os.walk(arg):
            if predicate(root):
                sub_directories[:] = []
                continue

            # NOTE(sigmavirus24): os.walk() will skip a directory if you
            # remove it from the list of sub-directories.
            for directory in sub_directories:
                joined = os.path.join(root, directory)
                if predicate(joined):
                    sub_directories.remove(directory)

            for filename in files:
                joined = os.path.join(root, filename)
                if not predicate(joined):
                    yield joined
    else:
        yield arg


def fnmatch(filename: str, patterns: Sequence[str]) -> bool:
    """Wrap :func:`fnmatch.fnmatch` to add some functionality.

    :param str filename:
        Name of the file we're trying to match.
    :param list patterns:
        Patterns we're using to try to match the filename.
    :param bool default:
        The default value if patterns is empty
    :returns:
        True if a pattern matches the filename, False if it doesn't.
        ``default`` if patterns is empty.
    """
    if not patterns:
        return True
    return any(_fnmatch.fnmatch(filename, pattern) for pattern in patterns)


def parameters_for(plugin: "Plugin") -> Dict[str, bool]:
    """Return the parameters for the plugin.

    This will inspect the plugin and return either the function parameters
    if the plugin is a function or the parameters for ``__init__`` after
    ``self`` if the plugin is a class.

    :param plugin:
        The internal plugin object.
    :type plugin:
        flake8.plugins.manager.Plugin
    :returns:
        A dictionary mapping the parameter name to whether or not it is
        required (a.k.a., is positional only/does not have a default).
    :rtype:
        dict([(str, bool)])
    """
    func = plugin.plugin
    is_class = not inspect.isfunction(func)
    if is_class:  # The plugin is a class
        func = plugin.plugin.__init__

    parameters = {
        parameter.name: parameter.default is parameter.empty
        for parameter in inspect.signature(func).parameters.values()
        if parameter.kind == parameter.POSITIONAL_OR_KEYWORD
    }

    if is_class:
        parameters.pop("self", None)

    return parameters


def matches_filename(
    path: str,
    patterns: Sequence[str],
    log_message: str,
    logger: logging.Logger,
) -> bool:
    """Use fnmatch to discern if a path exists in patterns.

    :param str path:
        The path to the file under question
    :param patterns:
        The patterns to match the path against.
    :type patterns:
        list[str]
    :param str log_message:
        The message used for logging purposes.
    :returns:
        True if path matches patterns, False otherwise
    :rtype:
        bool
    """
    if not patterns:
        return False
    basename = os.path.basename(path)
    if basename not in {".", ".."} and fnmatch(basename, patterns):
        logger.debug(log_message, {"path": basename, "whether": ""})
        return True

    absolute_path = os.path.abspath(path)
    match = fnmatch(absolute_path, patterns)
    logger.debug(
        log_message,
        {"path": absolute_path, "whether": "" if match else "not "},
    )
    return match


def get_python_version() -> str:
    """Find and format the python implementation and version.

    :returns:
        Implementation name, version, and platform as a string.
    :rtype:
        str
    """
    return "{} {} on {}".format(
        platform.python_implementation(),
        platform.python_version(),
        platform.system(),
    )
