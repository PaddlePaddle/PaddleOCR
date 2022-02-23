#!/usr/bin/env python
# pycodestyle.py - Check Python source code formatting, according to
# PEP 8
#
# Copyright (C) 2006-2009 Johann C. Rocholl <johann@rocholl.net>
# Copyright (C) 2009-2014 Florent Xicluna <florent.xicluna@gmail.com>
# Copyright (C) 2014-2016 Ian Lee <ianlee1521@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
r"""
Check Python source code formatting, according to PEP 8.

For usage and a list of options, try this:
$ python pycodestyle.py -h

This program and its regression test suite live here:
https://github.com/pycqa/pycodestyle

Groups of errors and warnings:
E errors
W warnings
100 indentation
200 whitespace
300 blank lines
400 imports
500 line length
600 deprecation
700 statements
900 syntax error
"""
from __future__ import with_statement

import bisect
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings

try:
    from functools import lru_cache
except ImportError:
    def lru_cache(maxsize=128):  # noqa as it's a fake implementation.
        """Does not really need a real a lru_cache, it's just
        optimization, so let's just do nothing here. Python 3.2+ will
        just get better performances, time to upgrade?
        """
        return lambda function: function

from fnmatch import fnmatch
from optparse import OptionParser

try:
    from configparser import RawConfigParser
    from io import TextIOWrapper
except ImportError:
    from ConfigParser import RawConfigParser

# this is a performance hack.  see https://bugs.python.org/issue43014
if (
        sys.version_info < (3, 10) and
        callable(getattr(tokenize, '_compile', None))
):  # pragma: no cover (<py310)
    tokenize._compile = lru_cache()(tokenize._compile)  # type: ignore

__version__ = '2.8.0'

DEFAULT_EXCLUDE = '.svn,CVS,.bzr,.hg,.git,__pycache__,.tox'
DEFAULT_IGNORE = 'E121,E123,E126,E226,E24,E704,W503,W504'
try:
    if sys.platform == 'win32':
        USER_CONFIG = os.path.expanduser(r'~\.pycodestyle')
    else:
        USER_CONFIG = os.path.join(
            os.getenv('XDG_CONFIG_HOME') or os.path.expanduser('~/.config'),
            'pycodestyle'
        )
except ImportError:
    USER_CONFIG = None

PROJECT_CONFIG = ('setup.cfg', 'tox.ini')
TESTSUITE_PATH = os.path.join(os.path.dirname(__file__), 'testsuite')
MAX_LINE_LENGTH = 79
# Number of blank lines between various code parts.
BLANK_LINES_CONFIG = {
    # Top level class and function.
    'top_level': 2,
    # Methods and nested class and function.
    'method': 1,
}
MAX_DOC_LENGTH = 72
INDENT_SIZE = 4
REPORT_FORMAT = {
    'default': '%(path)s:%(row)d:%(col)d: %(code)s %(text)s',
    'pylint': '%(path)s:%(row)d: [%(code)s] %(text)s',
}

PyCF_ONLY_AST = 1024
SINGLETONS = frozenset(['False', 'None', 'True'])
KEYWORDS = frozenset(keyword.kwlist + ['print', 'async']) - SINGLETONS
UNARY_OPERATORS = frozenset(['>>', '**', '*', '+', '-'])
ARITHMETIC_OP = frozenset(['**', '*', '/', '//', '+', '-', '@'])
WS_OPTIONAL_OPERATORS = ARITHMETIC_OP.union(['^', '&', '|', '<<', '>>', '%'])
# Warn for -> function annotation operator in py3.5+ (issue 803)
FUNCTION_RETURN_ANNOTATION_OP = ['->'] if sys.version_info >= (3, 5) else []
ASSIGNMENT_EXPRESSION_OP = [':='] if sys.version_info >= (3, 8) else []
WS_NEEDED_OPERATORS = frozenset([
    '**=', '*=', '/=', '//=', '+=', '-=', '!=', '<>', '<', '>',
    '%=', '^=', '&=', '|=', '==', '<=', '>=', '<<=', '>>=', '=',
    'and', 'in', 'is', 'or'] +
    FUNCTION_RETURN_ANNOTATION_OP +
    ASSIGNMENT_EXPRESSION_OP)
WHITESPACE = frozenset(' \t')
NEWLINE = frozenset([tokenize.NL, tokenize.NEWLINE])
SKIP_TOKENS = NEWLINE.union([tokenize.INDENT, tokenize.DEDENT])
# ERRORTOKEN is triggered by backticks in Python 3
SKIP_COMMENTS = SKIP_TOKENS.union([tokenize.COMMENT, tokenize.ERRORTOKEN])
BENCHMARK_KEYS = ['directories', 'files', 'logical lines', 'physical lines']

INDENT_REGEX = re.compile(r'([ \t]*)')
RAISE_COMMA_REGEX = re.compile(r'raise\s+\w+\s*,')
RERAISE_COMMA_REGEX = re.compile(r'raise\s+\w+\s*,.*,\s*\w+\s*$')
ERRORCODE_REGEX = re.compile(r'\b[A-Z]\d{3}\b')
DOCSTRING_REGEX = re.compile(r'u?r?["\']')
EXTRANEOUS_WHITESPACE_REGEX = re.compile(r'[\[({][ \t]|[ \t][\]}),;:](?!=)')
WHITESPACE_AFTER_COMMA_REGEX = re.compile(r'[,;:]\s*(?:  |\t)')
COMPARE_SINGLETON_REGEX = re.compile(r'(\bNone|\bFalse|\bTrue)?\s*([=!]=)'
                                     r'\s*(?(1)|(None|False|True))\b')
COMPARE_NEGATIVE_REGEX = re.compile(r'\b(?<!is\s)(not)\s+[^][)(}{ ]+\s+'
                                    r'(in|is)\s')
COMPARE_TYPE_REGEX = re.compile(r'(?:[=!]=|is(?:\s+not)?)\s+type(?:s.\w+Type'
                                r'|\s*\(\s*([^)]*[^ )])\s*\))')
KEYWORD_REGEX = re.compile(r'(\s*)\b(?:%s)\b(\s*)' % r'|'.join(KEYWORDS))
OPERATOR_REGEX = re.compile(r'(?:[^,\s])(\s*)(?:[-+*/|!<=>%&^]+)(\s*)')
LAMBDA_REGEX = re.compile(r'\blambda\b')
HUNK_REGEX = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@.*$')
STARTSWITH_DEF_REGEX = re.compile(r'^(async\s+def|def)\b')
STARTSWITH_TOP_LEVEL_REGEX = re.compile(r'^(async\s+def\s+|def\s+|class\s+|@)')
STARTSWITH_INDENT_STATEMENT_REGEX = re.compile(
    r'^\s*({0})\b'.format('|'.join(s.replace(' ', r'\s+') for s in (
        'def', 'async def',
        'for', 'async for',
        'if', 'elif', 'else',
        'try', 'except', 'finally',
        'with', 'async with',
        'class',
        'while',
    )))
)
DUNDER_REGEX = re.compile(r"^__([^\s]+)__(?::\s*[a-zA-Z.0-9_\[\]\"]+)? = ")
BLANK_EXCEPT_REGEX = re.compile(r"except\s*:")

_checks = {'physical_line': {}, 'logical_line': {}, 'tree': {}}


def _get_parameters(function):
    if sys.version_info >= (3, 3):
        return [parameter.name
                for parameter
                in inspect.signature(function).parameters.values()
                if parameter.kind == parameter.POSITIONAL_OR_KEYWORD]
    else:
        return inspect.getargspec(function)[0]


def register_check(check, codes=None):
    """Register a new check object."""
    def _add_check(check, kind, codes, args):
        if check in _checks[kind]:
            _checks[kind][check][0].extend(codes or [])
        else:
            _checks[kind][check] = (codes or [''], args)
    if inspect.isfunction(check):
        args = _get_parameters(check)
        if args and args[0] in ('physical_line', 'logical_line'):
            if codes is None:
                codes = ERRORCODE_REGEX.findall(check.__doc__ or '')
            _add_check(check, args[0], codes, args)
    elif inspect.isclass(check):
        if _get_parameters(check.__init__)[:2] == ['self', 'tree']:
            _add_check(check, 'tree', codes, None)
    return check


########################################################################
# Plugins (check functions) for physical lines
########################################################################

@register_check
def tabs_or_spaces(physical_line, indent_char):
    r"""Never mix tabs and spaces.

    The most popular way of indenting Python is with spaces only.  The
    second-most popular way is with tabs only.  Code indented with a
    mixture of tabs and spaces should be converted to using spaces
    exclusively.  When invoking the Python command line interpreter with
    the -t option, it issues warnings about code that illegally mixes
    tabs and spaces.  When using -tt these warnings become errors.
    These options are highly recommended!

    Okay: if a == 0:\n    a = 1\n    b = 1
    E101: if a == 0:\n        a = 1\n\tb = 1
    """
    indent = INDENT_REGEX.match(physical_line).group(1)
    for offset, char in enumerate(indent):
        if char != indent_char:
            return offset, "E101 indentation contains mixed spaces and tabs"


@register_check
def tabs_obsolete(physical_line):
    r"""On new projects, spaces-only are strongly recommended over tabs.

    Okay: if True:\n    return
    W191: if True:\n\treturn
    """
    indent = INDENT_REGEX.match(physical_line).group(1)
    if '\t' in indent:
        return indent.index('\t'), "W191 indentation contains tabs"


@register_check
def trailing_whitespace(physical_line):
    r"""Trailing whitespace is superfluous.

    The warning returned varies on whether the line itself is blank,
    for easier filtering for those who want to indent their blank lines.

    Okay: spam(1)\n#
    W291: spam(1) \n#
    W293: class Foo(object):\n    \n    bang = 12
    """
    physical_line = physical_line.rstrip('\n')    # chr(10), newline
    physical_line = physical_line.rstrip('\r')    # chr(13), carriage return
    physical_line = physical_line.rstrip('\x0c')  # chr(12), form feed, ^L
    stripped = physical_line.rstrip(' \t\v')
    if physical_line != stripped:
        if stripped:
            return len(stripped), "W291 trailing whitespace"
        else:
            return 0, "W293 blank line contains whitespace"


@register_check
def trailing_blank_lines(physical_line, lines, line_number, total_lines):
    r"""Trailing blank lines are superfluous.

    Okay: spam(1)
    W391: spam(1)\n

    However the last line should end with a new line (warning W292).
    """
    if line_number == total_lines:
        stripped_last_line = physical_line.rstrip('\r\n')
        if physical_line and not stripped_last_line:
            return 0, "W391 blank line at end of file"
        if stripped_last_line == physical_line:
            return len(lines[-1]), "W292 no newline at end of file"


@register_check
def maximum_line_length(physical_line, max_line_length, multiline,
                        line_number, noqa):
    r"""Limit all lines to a maximum of 79 characters.

    There are still many devices around that are limited to 80 character
    lines; plus, limiting windows to 80 characters makes it possible to
    have several windows side-by-side.  The default wrapping on such
    devices looks ugly.  Therefore, please limit all lines to a maximum
    of 79 characters. For flowing long blocks of text (docstrings or
    comments), limiting the length to 72 characters is recommended.

    Reports error E501.
    """
    line = physical_line.rstrip()
    length = len(line)
    if length > max_line_length and not noqa:
        # Special case: ignore long shebang lines.
        if line_number == 1 and line.startswith('#!'):
            return
        # Special case for long URLs in multi-line docstrings or
        # comments, but still report the error when the 72 first chars
        # are whitespaces.
        chunks = line.split()
        if ((len(chunks) == 1 and multiline) or
            (len(chunks) == 2 and chunks[0] == '#')) and \
                len(line) - len(chunks[-1]) < max_line_length - 7:
            return
        if hasattr(line, 'decode'):   # Python 2
            # The line could contain multi-byte characters
            try:
                length = len(line.decode('utf-8'))
            except UnicodeError:
                pass
        if length > max_line_length:
            return (max_line_length, "E501 line too long "
                    "(%d > %d characters)" % (length, max_line_length))


########################################################################
# Plugins (check functions) for logical lines
########################################################################


def _is_one_liner(logical_line, indent_level, lines, line_number):
    if not STARTSWITH_TOP_LEVEL_REGEX.match(logical_line):
        return False

    line_idx = line_number - 1

    if line_idx < 1:
        prev_indent = 0
    else:
        prev_indent = expand_indent(lines[line_idx - 1])

    if prev_indent > indent_level:
        return False

    while line_idx < len(lines):
        line = lines[line_idx].strip()
        if not line.startswith('@') and STARTSWITH_TOP_LEVEL_REGEX.match(line):
            break
        else:
            line_idx += 1
    else:
        return False  # invalid syntax: EOF while searching for def/class

    next_idx = line_idx + 1
    while next_idx < len(lines):
        if lines[next_idx].strip():
            break
        else:
            next_idx += 1
    else:
        return True  # line is last in the file

    return expand_indent(lines[next_idx]) <= indent_level


@register_check
def blank_lines(logical_line, blank_lines, indent_level, line_number,
                blank_before, previous_logical,
                previous_unindented_logical_line, previous_indent_level,
                lines):
    r"""Separate top-level function and class definitions with two blank
    lines.

    Method definitions inside a class are separated by a single blank
    line.

    Extra blank lines may be used (sparingly) to separate groups of
    related functions.  Blank lines may be omitted between a bunch of
    related one-liners (e.g. a set of dummy implementations).

    Use blank lines in functions, sparingly, to indicate logical
    sections.

    Okay: def a():\n    pass\n\n\ndef b():\n    pass
    Okay: def a():\n    pass\n\n\nasync def b():\n    pass
    Okay: def a():\n    pass\n\n\n# Foo\n# Bar\n\ndef b():\n    pass
    Okay: default = 1\nfoo = 1
    Okay: classify = 1\nfoo = 1

    E301: class Foo:\n    b = 0\n    def bar():\n        pass
    E302: def a():\n    pass\n\ndef b(n):\n    pass
    E302: def a():\n    pass\n\nasync def b(n):\n    pass
    E303: def a():\n    pass\n\n\n\ndef b(n):\n    pass
    E303: def a():\n\n\n\n    pass
    E304: @decorator\n\ndef a():\n    pass
    E305: def a():\n    pass\na()
    E306: def a():\n    def b():\n        pass\n    def c():\n        pass
    """  # noqa
    top_level_lines = BLANK_LINES_CONFIG['top_level']
    method_lines = BLANK_LINES_CONFIG['method']

    if not previous_logical and blank_before < top_level_lines:
        return  # Don't expect blank lines before the first line
    if previous_logical.startswith('@'):
        if blank_lines:
            yield 0, "E304 blank lines found after function decorator"
    elif (blank_lines > top_level_lines or
            (indent_level and blank_lines == method_lines + 1)
          ):
        yield 0, "E303 too many blank lines (%d)" % blank_lines
    elif STARTSWITH_TOP_LEVEL_REGEX.match(logical_line):
        # allow a group of one-liners
        if (
            _is_one_liner(logical_line, indent_level, lines, line_number) and
            blank_before == 0
        ):
            return
        if indent_level:
            if not (blank_before == method_lines or
                    previous_indent_level < indent_level or
                    DOCSTRING_REGEX.match(previous_logical)
                    ):
                ancestor_level = indent_level
                nested = False
                # Search backwards for a def ancestor or tree root
                # (top level).
                for line in lines[line_number - top_level_lines::-1]:
                    if line.strip() and expand_indent(line) < ancestor_level:
                        ancestor_level = expand_indent(line)
                        nested = STARTSWITH_DEF_REGEX.match(line.lstrip())
                        if nested or ancestor_level == 0:
                            break
                if nested:
                    yield 0, "E306 expected %s blank line before a " \
                        "nested definition, found 0" % (method_lines,)
                else:
                    yield 0, "E301 expected %s blank line, found 0" % (
                        method_lines,)
        elif blank_before != top_level_lines:
            yield 0, "E302 expected %s blank lines, found %d" % (
                top_level_lines, blank_before)
    elif (logical_line and
            not indent_level and
            blank_before != top_level_lines and
            previous_unindented_logical_line.startswith(('def ', 'class '))
          ):
        yield 0, "E305 expected %s blank lines after " \
            "class or function definition, found %d" % (
                top_level_lines, blank_before)


@register_check
def extraneous_whitespace(logical_line):
    r"""Avoid extraneous whitespace.

    Avoid extraneous whitespace in these situations:
    - Immediately inside parentheses, brackets or braces.
    - Immediately before a comma, semicolon, or colon.

    Okay: spam(ham[1], {eggs: 2})
    E201: spam( ham[1], {eggs: 2})
    E201: spam(ham[ 1], {eggs: 2})
    E201: spam(ham[1], { eggs: 2})
    E202: spam(ham[1], {eggs: 2} )
    E202: spam(ham[1 ], {eggs: 2})
    E202: spam(ham[1], {eggs: 2 })

    E203: if x == 4: print x, y; x, y = y , x
    E203: if x == 4: print x, y ; x, y = y, x
    E203: if x == 4 : print x, y; x, y = y, x
    """
    line = logical_line
    for match in EXTRANEOUS_WHITESPACE_REGEX.finditer(line):
        text = match.group()
        char = text.strip()
        found = match.start()
        if text[-1].isspace():
            # assert char in '([{'
            yield found + 1, "E201 whitespace after '%s'" % char
        elif line[found - 1] != ',':
            code = ('E202' if char in '}])' else 'E203')  # if char in ',;:'
            yield found, "%s whitespace before '%s'" % (code, char)


@register_check
def whitespace_around_keywords(logical_line):
    r"""Avoid extraneous whitespace around keywords.

    Okay: True and False
    E271: True and  False
    E272: True  and False
    E273: True and\tFalse
    E274: True\tand False
    """
    for match in KEYWORD_REGEX.finditer(logical_line):
        before, after = match.groups()

        if '\t' in before:
            yield match.start(1), "E274 tab before keyword"
        elif len(before) > 1:
            yield match.start(1), "E272 multiple spaces before keyword"

        if '\t' in after:
            yield match.start(2), "E273 tab after keyword"
        elif len(after) > 1:
            yield match.start(2), "E271 multiple spaces after keyword"


@register_check
def missing_whitespace_after_import_keyword(logical_line):
    r"""Multiple imports in form from x import (a, b, c) should have
    space between import statement and parenthesised name list.

    Okay: from foo import (bar, baz)
    E275: from foo import(bar, baz)
    E275: from importable.module import(bar, baz)
    """
    line = logical_line
    indicator = ' import('
    if line.startswith('from '):
        found = line.find(indicator)
        if -1 < found:
            pos = found + len(indicator) - 1
            yield pos, "E275 missing whitespace after keyword"


@register_check
def missing_whitespace(logical_line):
    r"""Each comma, semicolon or colon should be followed by whitespace.

    Okay: [a, b]
    Okay: (3,)
    Okay: a[1:4]
    Okay: a[:4]
    Okay: a[1:]
    Okay: a[1:4:2]
    E231: ['a','b']
    E231: foo(bar,baz)
    E231: [{'a':'b'}]
    """
    line = logical_line
    for index in range(len(line) - 1):
        char = line[index]
        next_char = line[index + 1]
        if char in ',;:' and next_char not in WHITESPACE:
            before = line[:index]
            if char == ':' and before.count('[') > before.count(']') and \
                    before.rfind('{') < before.rfind('['):
                continue  # Slice syntax, no space required
            if char == ',' and next_char == ')':
                continue  # Allow tuple with only one element: (3,)
            if char == ':' and next_char == '=' and sys.version_info >= (3, 8):
                continue  # Allow assignment expression
            yield index, "E231 missing whitespace after '%s'" % char


@register_check
def indentation(logical_line, previous_logical, indent_char,
                indent_level, previous_indent_level,
                indent_size):
    r"""Use indent_size (PEP8 says 4) spaces per indentation level.

    For really old code that you don't want to mess up, you can continue
    to use 8-space tabs.

    Okay: a = 1
    Okay: if a == 0:\n    a = 1
    E111:   a = 1
    E114:   # a = 1

    Okay: for item in items:\n    pass
    E112: for item in items:\npass
    E115: for item in items:\n# Hi\n    pass

    Okay: a = 1\nb = 2
    E113: a = 1\n    b = 2
    E116: a = 1\n    # b = 2
    """
    c = 0 if logical_line else 3
    tmpl = "E11%d %s" if logical_line else "E11%d %s (comment)"
    if indent_level % indent_size:
        yield 0, tmpl % (
            1 + c,
            "indentation is not a multiple of " + str(indent_size),
        )
    indent_expect = previous_logical.endswith(':')
    if indent_expect and indent_level <= previous_indent_level:
        yield 0, tmpl % (2 + c, "expected an indented block")
    elif not indent_expect and indent_level > previous_indent_level:
        yield 0, tmpl % (3 + c, "unexpected indentation")

    if indent_expect:
        expected_indent_amount = 8 if indent_char == '\t' else 4
        expected_indent_level = previous_indent_level + expected_indent_amount
        if indent_level > expected_indent_level:
            yield 0, tmpl % (7, 'over-indented')


@register_check
def continued_indentation(logical_line, tokens, indent_level, hang_closing,
                          indent_char, indent_size, noqa, verbose):
    r"""Continuation lines indentation.

    Continuation lines should align wrapped elements either vertically
    using Python's implicit line joining inside parentheses, brackets
    and braces, or using a hanging indent.

    When using a hanging indent these considerations should be applied:
    - there should be no arguments on the first line, and
    - further indentation should be used to clearly distinguish itself
      as a continuation line.

    Okay: a = (\n)
    E123: a = (\n    )

    Okay: a = (\n    42)
    E121: a = (\n   42)
    E122: a = (\n42)
    E123: a = (\n    42\n    )
    E124: a = (24,\n     42\n)
    E125: if (\n    b):\n    pass
    E126: a = (\n        42)
    E127: a = (24,\n      42)
    E128: a = (24,\n    42)
    E129: if (a or\n    b):\n    pass
    E131: a = (\n    42\n 24)
    """
    first_row = tokens[0][2][0]
    nrows = 1 + tokens[-1][2][0] - first_row
    if noqa or nrows == 1:
        return

    # indent_next tells us whether the next block is indented; assuming
    # that it is indented by 4 spaces, then we should not allow 4-space
    # indents on the final continuation line; in turn, some other
    # indents are allowed to have an extra 4 spaces.
    indent_next = logical_line.endswith(':')

    row = depth = 0
    valid_hangs = (indent_size,) if indent_char != '\t' \
        else (indent_size, indent_size * 2)
    # remember how many brackets were opened on each line
    parens = [0] * nrows
    # relative indents of physical lines
    rel_indent = [0] * nrows
    # for each depth, collect a list of opening rows
    open_rows = [[0]]
    # for each depth, memorize the hanging indentation
    hangs = [None]
    # visual indents
    indent_chances = {}
    last_indent = tokens[0][2]
    visual_indent = None
    last_token_multiline = False
    # for each depth, memorize the visual indent column
    indent = [last_indent[1]]
    if verbose >= 3:
        print(">>> " + tokens[0][4].rstrip())

    for token_type, text, start, end, line in tokens:

        newline = row < start[0] - first_row
        if newline:
            row = start[0] - first_row
            newline = not last_token_multiline and token_type not in NEWLINE

        if newline:
            # this is the beginning of a continuation line.
            last_indent = start
            if verbose >= 3:
                print("... " + line.rstrip())

            # record the initial indent.
            rel_indent[row] = expand_indent(line) - indent_level

            # identify closing bracket
            close_bracket = (token_type == tokenize.OP and text in ']})')

            # is the indent relative to an opening bracket line?
            for open_row in reversed(open_rows[depth]):
                hang = rel_indent[row] - rel_indent[open_row]
                hanging_indent = hang in valid_hangs
                if hanging_indent:
                    break
            if hangs[depth]:
                hanging_indent = (hang == hangs[depth])
            # is there any chance of visual indent?
            visual_indent = (not close_bracket and hang > 0 and
                             indent_chances.get(start[1]))

            if close_bracket and indent[depth]:
                # closing bracket for visual indent
                if start[1] != indent[depth]:
                    yield (start, "E124 closing bracket does not match "
                           "visual indentation")
            elif close_bracket and not hang:
                # closing bracket matches indentation of opening
                # bracket's line
                if hang_closing:
                    yield start, "E133 closing bracket is missing indentation"
            elif indent[depth] and start[1] < indent[depth]:
                if visual_indent is not True:
                    # visual indent is broken
                    yield (start, "E128 continuation line "
                           "under-indented for visual indent")
            elif hanging_indent or (indent_next and
                                    rel_indent[row] == 2 * indent_size):
                # hanging indent is verified
                if close_bracket and not hang_closing:
                    yield (start, "E123 closing bracket does not match "
                           "indentation of opening bracket's line")
                hangs[depth] = hang
            elif visual_indent is True:
                # visual indent is verified
                indent[depth] = start[1]
            elif visual_indent in (text, str):
                # ignore token lined up with matching one from a
                # previous line
                pass
            else:
                # indent is broken
                if hang <= 0:
                    error = "E122", "missing indentation or outdented"
                elif indent[depth]:
                    error = "E127", "over-indented for visual indent"
                elif not close_bracket and hangs[depth]:
                    error = "E131", "unaligned for hanging indent"
                else:
                    hangs[depth] = hang
                    if hang > indent_size:
                        error = "E126", "over-indented for hanging indent"
                    else:
                        error = "E121", "under-indented for hanging indent"
                yield start, "%s continuation line %s" % error

        # look for visual indenting
        if (parens[row] and
                token_type not in (tokenize.NL, tokenize.COMMENT) and
                not indent[depth]):
            indent[depth] = start[1]
            indent_chances[start[1]] = True
            if verbose >= 4:
                print("bracket depth %s indent to %s" % (depth, start[1]))
        # deal with implicit string concatenation
        elif (token_type in (tokenize.STRING, tokenize.COMMENT) or
              text in ('u', 'ur', 'b', 'br')):
            indent_chances[start[1]] = str
        # visual indent after assert/raise/with
        elif not row and not depth and text in ["assert", "raise", "with"]:
            indent_chances[end[1] + 1] = True
        # special case for the "if" statement because len("if (") == 4
        elif not indent_chances and not row and not depth and text == 'if':
            indent_chances[end[1] + 1] = True
        elif text == ':' and line[end[1]:].isspace():
            open_rows[depth].append(row)

        # keep track of bracket depth
        if token_type == tokenize.OP:
            if text in '([{':
                depth += 1
                indent.append(0)
                hangs.append(None)
                if len(open_rows) == depth:
                    open_rows.append([])
                open_rows[depth].append(row)
                parens[row] += 1
                if verbose >= 4:
                    print("bracket depth %s seen, col %s, visual min = %s" %
                          (depth, start[1], indent[depth]))
            elif text in ')]}' and depth > 0:
                # parent indents should not be more than this one
                prev_indent = indent.pop() or last_indent[1]
                hangs.pop()
                for d in range(depth):
                    if indent[d] > prev_indent:
                        indent[d] = 0
                for ind in list(indent_chances):
                    if ind >= prev_indent:
                        del indent_chances[ind]
                del open_rows[depth + 1:]
                depth -= 1
                if depth:
                    indent_chances[indent[depth]] = True
                for idx in range(row, -1, -1):
                    if parens[idx]:
                        parens[idx] -= 1
                        break
            assert len(indent) == depth + 1
            if start[1] not in indent_chances:
                # allow lining up tokens
                indent_chances[start[1]] = text

        last_token_multiline = (start[0] != end[0])
        if last_token_multiline:
            rel_indent[end[0] - first_row] = rel_indent[row]

    if indent_next and expand_indent(line) == indent_level + indent_size:
        pos = (start[0], indent[0] + indent_size)
        if visual_indent:
            code = "E129 visually indented line"
        else:
            code = "E125 continuation line"
        yield pos, "%s with same indent as next logical line" % code


@register_check
def whitespace_before_parameters(logical_line, tokens):
    r"""Avoid extraneous whitespace.

    Avoid extraneous whitespace in the following situations:
    - before the open parenthesis that starts the argument list of a
      function call.
    - before the open parenthesis that starts an indexing or slicing.

    Okay: spam(1)
    E211: spam (1)

    Okay: dict['key'] = list[index]
    E211: dict ['key'] = list[index]
    E211: dict['key'] = list [index]
    """
    prev_type, prev_text, __, prev_end, __ = tokens[0]
    for index in range(1, len(tokens)):
        token_type, text, start, end, __ = tokens[index]
        if (
            token_type == tokenize.OP and
            text in '([' and
            start != prev_end and
            (prev_type == tokenize.NAME or prev_text in '}])') and
            # Syntax "class A (B):" is allowed, but avoid it
            (index < 2 or tokens[index - 2][1] != 'class') and
            # Allow "return (a.foo for a in range(5))"
            not keyword.iskeyword(prev_text) and
            # 'match' and 'case' are only soft keywords
            (
                sys.version_info < (3, 9) or
                not keyword.issoftkeyword(prev_text)
            )
        ):
            yield prev_end, "E211 whitespace before '%s'" % text
        prev_type = token_type
        prev_text = text
        prev_end = end


@register_check
def whitespace_around_operator(logical_line):
    r"""Avoid extraneous whitespace around an operator.

    Okay: a = 12 + 3
    E221: a = 4  + 5
    E222: a = 4 +  5
    E223: a = 4\t+ 5
    E224: a = 4 +\t5
    """
    for match in OPERATOR_REGEX.finditer(logical_line):
        before, after = match.groups()

        if '\t' in before:
            yield match.start(1), "E223 tab before operator"
        elif len(before) > 1:
            yield match.start(1), "E221 multiple spaces before operator"

        if '\t' in after:
            yield match.start(2), "E224 tab after operator"
        elif len(after) > 1:
            yield match.start(2), "E222 multiple spaces after operator"


@register_check
def missing_whitespace_around_operator(logical_line, tokens):
    r"""Surround operators with a single space on either side.

    - Always surround these binary operators with a single space on
      either side: assignment (=), augmented assignment (+=, -= etc.),
      comparisons (==, <, >, !=, <=, >=, in, not in, is, is not),
      Booleans (and, or, not).

    - If operators with different priorities are used, consider adding
      whitespace around the operators with the lowest priorities.

    Okay: i = i + 1
    Okay: submitted += 1
    Okay: x = x * 2 - 1
    Okay: hypot2 = x * x + y * y
    Okay: c = (a + b) * (a - b)
    Okay: foo(bar, key='word', *args, **kwargs)
    Okay: alpha[:-i]

    E225: i=i+1
    E225: submitted +=1
    E225: x = x /2 - 1
    E225: z = x **y
    E225: z = 1and 1
    E226: c = (a+b) * (a-b)
    E226: hypot2 = x*x + y*y
    E227: c = a|b
    E228: msg = fmt%(errno, errmsg)
    """
    parens = 0
    need_space = False
    prev_type = tokenize.OP
    prev_text = prev_end = None
    operator_types = (tokenize.OP, tokenize.NAME)
    for token_type, text, start, end, line in tokens:
        if token_type in SKIP_COMMENTS:
            continue
        if text in ('(', 'lambda'):
            parens += 1
        elif text == ')':
            parens -= 1
        if need_space:
            if start != prev_end:
                # Found a (probably) needed space
                if need_space is not True and not need_space[1]:
                    yield (need_space[0],
                           "E225 missing whitespace around operator")
                need_space = False
            elif text == '>' and prev_text in ('<', '-'):
                # Tolerate the "<>" operator, even if running Python 3
                # Deal with Python 3's annotated return value "->"
                pass
            elif (
                    # def f(a, /, b):
                    #           ^
                    # def f(a, b, /):
                    #              ^
                    # f = lambda a, /:
                    #                ^
                    prev_text == '/' and text in {',', ')', ':'} or
                    # def f(a, b, /):
                    #               ^
                    prev_text == ')' and text == ':'
            ):
                # Tolerate the "/" operator in function definition
                # For more info see PEP570
                pass
            else:
                if need_space is True or need_space[1]:
                    # A needed trailing space was not found
                    yield prev_end, "E225 missing whitespace around operator"
                elif prev_text != '**':
                    code, optype = 'E226', 'arithmetic'
                    if prev_text == '%':
                        code, optype = 'E228', 'modulo'
                    elif prev_text not in ARITHMETIC_OP:
                        code, optype = 'E227', 'bitwise or shift'
                    yield (need_space[0], "%s missing whitespace "
                           "around %s operator" % (code, optype))
                need_space = False
        elif token_type in operator_types and prev_end is not None:
            if text == '=' and parens:
                # Allow keyword args or defaults: foo(bar=None).
                pass
            elif text in WS_NEEDED_OPERATORS:
                need_space = True
            elif text in UNARY_OPERATORS:
                # Check if the operator is used as a binary operator
                # Allow unary operators: -123, -x, +1.
                # Allow argument unpacking: foo(*args, **kwargs).
                if prev_type == tokenize.OP and prev_text in '}])' or (
                    prev_type != tokenize.OP and
                    prev_text not in KEYWORDS and (
                        sys.version_info < (3, 9) or
                        not keyword.issoftkeyword(prev_text)
                    )
                ):
                    need_space = None
            elif text in WS_OPTIONAL_OPERATORS:
                need_space = None

            if need_space is None:
                # Surrounding space is optional, but ensure that
                # trailing space matches opening space
                need_space = (prev_end, start != prev_end)
            elif need_space and start == prev_end:
                # A needed opening space was not found
                yield prev_end, "E225 missing whitespace around operator"
                need_space = False
        prev_type = token_type
        prev_text = text
        prev_end = end


@register_check
def whitespace_around_comma(logical_line):
    r"""Avoid extraneous whitespace after a comma or a colon.

    Note: these checks are disabled by default

    Okay: a = (1, 2)
    E241: a = (1,  2)
    E242: a = (1,\t2)
    """
    line = logical_line
    for m in WHITESPACE_AFTER_COMMA_REGEX.finditer(line):
        found = m.start() + 1
        if '\t' in m.group():
            yield found, "E242 tab after '%s'" % m.group()[0]
        else:
            yield found, "E241 multiple spaces after '%s'" % m.group()[0]


@register_check
def whitespace_around_named_parameter_equals(logical_line, tokens):
    r"""Don't use spaces around the '=' sign in function arguments.

    Don't use spaces around the '=' sign when used to indicate a
    keyword argument or a default parameter value, except when
    using a type annotation.

    Okay: def complex(real, imag=0.0):
    Okay: return magic(r=real, i=imag)
    Okay: boolean(a == b)
    Okay: boolean(a != b)
    Okay: boolean(a <= b)
    Okay: boolean(a >= b)
    Okay: def foo(arg: int = 42):
    Okay: async def foo(arg: int = 42):

    E251: def complex(real, imag = 0.0):
    E251: return magic(r = real, i = imag)
    E252: def complex(real, image: float=0.0):
    """
    parens = 0
    no_space = False
    require_space = False
    prev_end = None
    annotated_func_arg = False
    in_def = bool(STARTSWITH_DEF_REGEX.match(logical_line))

    message = "E251 unexpected spaces around keyword / parameter equals"
    missing_message = "E252 missing whitespace around parameter equals"

    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.NL:
            continue
        if no_space:
            no_space = False
            if start != prev_end:
                yield (prev_end, message)
        if require_space:
            require_space = False
            if start == prev_end:
                yield (prev_end, missing_message)
        if token_type == tokenize.OP:
            if text in '([':
                parens += 1
            elif text in ')]':
                parens -= 1
            elif in_def and text == ':' and parens == 1:
                annotated_func_arg = True
            elif parens == 1 and text == ',':
                annotated_func_arg = False
            elif parens and text == '=':
                if annotated_func_arg and parens == 1:
                    require_space = True
                    if start == prev_end:
                        yield (prev_end, missing_message)
                else:
                    no_space = True
                    if start != prev_end:
                        yield (prev_end, message)
            if not parens:
                annotated_func_arg = False

        prev_end = end


@register_check
def whitespace_before_comment(logical_line, tokens):
    r"""Separate inline comments by at least two spaces.

    An inline comment is a comment on the same line as a statement.
    Inline comments should be separated by at least two spaces from the
    statement. They should start with a # and a single space.

    Each line of a block comment starts with a # and a single space
    (unless it is indented text inside the comment).

    Okay: x = x + 1  # Increment x
    Okay: x = x + 1    # Increment x
    Okay: # Block comment
    E261: x = x + 1 # Increment x
    E262: x = x + 1  #Increment x
    E262: x = x + 1  #  Increment x
    E265: #Block comment
    E266: ### Block comment
    """
    prev_end = (0, 0)
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            inline_comment = line[:start[1]].strip()
            if inline_comment:
                if prev_end[0] == start[0] and start[1] < prev_end[1] + 2:
                    yield (prev_end,
                           "E261 at least two spaces before inline comment")
            symbol, sp, comment = text.partition(' ')
            bad_prefix = symbol not in '#:' and (symbol.lstrip('#')[:1] or '#')
            if inline_comment:
                if bad_prefix or comment[:1] in WHITESPACE:
                    yield start, "E262 inline comment should start with '# '"
            elif bad_prefix and (bad_prefix != '!' or start[0] > 1):
                if bad_prefix != '#':
                    yield start, "E265 block comment should start with '# '"
                elif comment:
                    yield start, "E266 too many leading '#' for block comment"
        elif token_type != tokenize.NL:
            prev_end = end


@register_check
def imports_on_separate_lines(logical_line):
    r"""Place imports on separate lines.

    Okay: import os\nimport sys
    E401: import sys, os

    Okay: from subprocess import Popen, PIPE
    Okay: from myclas import MyClass
    Okay: from foo.bar.yourclass import YourClass
    Okay: import myclass
    Okay: import foo.bar.yourclass
    """
    line = logical_line
    if line.startswith('import '):
        found = line.find(',')
        if -1 < found and ';' not in line[:found]:
            yield found, "E401 multiple imports on one line"


@register_check
def module_imports_on_top_of_file(
        logical_line, indent_level, checker_state, noqa):
    r"""Place imports at the top of the file.

    Always put imports at the top of the file, just after any module
    comments and docstrings, and before module globals and constants.

    Okay: import os
    Okay: # this is a comment\nimport os
    Okay: '''this is a module docstring'''\nimport os
    Okay: r'''this is a module docstring'''\nimport os
    Okay:
    try:\n\timport x\nexcept ImportError:\n\tpass\nelse:\n\tpass\nimport y
    Okay:
    try:\n\timport x\nexcept ImportError:\n\tpass\nfinally:\n\tpass\nimport y
    E402: a=1\nimport os
    E402: 'One string'\n"Two string"\nimport os
    E402: a=1\nfrom sys import x

    Okay: if x:\n    import os
    """  # noqa
    def is_string_literal(line):
        if line[0] in 'uUbB':
            line = line[1:]
        if line and line[0] in 'rR':
            line = line[1:]
        return line and (line[0] == '"' or line[0] == "'")

    allowed_keywords = (
        'try', 'except', 'else', 'finally', 'with', 'if', 'elif')

    if indent_level:  # Allow imports in conditional statement/function
        return
    if not logical_line:  # Allow empty lines or comments
        return
    if noqa:
        return
    line = logical_line
    if line.startswith('import ') or line.startswith('from '):
        if checker_state.get('seen_non_imports', False):
            yield 0, "E402 module level import not at top of file"
    elif re.match(DUNDER_REGEX, line):
        return
    elif any(line.startswith(kw) for kw in allowed_keywords):
        # Allow certain keywords intermixed with imports in order to
        # support conditional or filtered importing
        return
    elif is_string_literal(line):
        # The first literal is a docstring, allow it. Otherwise, report
        # error.
        if checker_state.get('seen_docstring', False):
            checker_state['seen_non_imports'] = True
        else:
            checker_state['seen_docstring'] = True
    else:
        checker_state['seen_non_imports'] = True


@register_check
def compound_statements(logical_line):
    r"""Compound statements (on the same line) are generally
    discouraged.

    While sometimes it's okay to put an if/for/while with a small body
    on the same line, never do this for multi-clause statements.
    Also avoid folding such long lines!

    Always use a def statement instead of an assignment statement that
    binds a lambda expression directly to a name.

    Okay: if foo == 'blah':\n    do_blah_thing()
    Okay: do_one()
    Okay: do_two()
    Okay: do_three()

    E701: if foo == 'blah': do_blah_thing()
    E701: for x in lst: total += x
    E701: while t < 10: t = delay()
    E701: if foo == 'blah': do_blah_thing()
    E701: else: do_non_blah_thing()
    E701: try: something()
    E701: finally: cleanup()
    E701: if foo == 'blah': one(); two(); three()
    E702: do_one(); do_two(); do_three()
    E703: do_four();  # useless semicolon
    E704: def f(x): return 2*x
    E731: f = lambda x: 2*x
    """
    line = logical_line
    last_char = len(line) - 1
    found = line.find(':')
    prev_found = 0
    counts = {char: 0 for char in '{}[]()'}
    while -1 < found < last_char:
        update_counts(line[prev_found:found], counts)
        if ((counts['{'] <= counts['}'] and   # {'a': 1} (dict)
             counts['['] <= counts[']'] and   # [1:2] (slice)
             counts['('] <= counts[')']) and  # (annotation)
            not (sys.version_info >= (3, 8) and
                 line[found + 1] == '=')):  # assignment expression
            lambda_kw = LAMBDA_REGEX.search(line, 0, found)
            if lambda_kw:
                before = line[:lambda_kw.start()].rstrip()
                if before[-1:] == '=' and isidentifier(before[:-1].strip()):
                    yield 0, ("E731 do not assign a lambda expression, use a "
                              "def")
                break
            if STARTSWITH_DEF_REGEX.match(line):
                yield 0, "E704 multiple statements on one line (def)"
            elif STARTSWITH_INDENT_STATEMENT_REGEX.match(line):
                yield found, "E701 multiple statements on one line (colon)"
        prev_found = found
        found = line.find(':', found + 1)
    found = line.find(';')
    while -1 < found:
        if found < last_char:
            yield found, "E702 multiple statements on one line (semicolon)"
        else:
            yield found, "E703 statement ends with a semicolon"
        found = line.find(';', found + 1)


@register_check
def explicit_line_join(logical_line, tokens):
    r"""Avoid explicit line join between brackets.

    The preferred way of wrapping long lines is by using Python's
    implied line continuation inside parentheses, brackets and braces.
    Long lines can be broken over multiple lines by wrapping expressions
    in parentheses.  These should be used in preference to using a
    backslash for line continuation.

    E502: aaa = [123, \\n       123]
    E502: aaa = ("bbb " \\n       "ccc")

    Okay: aaa = [123,\n       123]
    Okay: aaa = ("bbb "\n       "ccc")
    Okay: aaa = "bbb " \\n    "ccc"
    Okay: aaa = 123  # \\
    """
    prev_start = prev_end = parens = 0
    comment = False
    backslash = None
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            comment = True
        if start[0] != prev_start and parens and backslash and not comment:
            yield backslash, "E502 the backslash is redundant between brackets"
        if end[0] != prev_end:
            if line.rstrip('\r\n').endswith('\\'):
                backslash = (end[0], len(line.splitlines()[-1]) - 1)
            else:
                backslash = None
            prev_start = prev_end = end[0]
        else:
            prev_start = start[0]
        if token_type == tokenize.OP:
            if text in '([{':
                parens += 1
            elif text in ')]}':
                parens -= 1


_SYMBOLIC_OPS = frozenset("()[]{},:.;@=%~") | frozenset(("...",))


def _is_binary_operator(token_type, text):
    is_op_token = token_type == tokenize.OP
    is_conjunction = text in ['and', 'or']
    # NOTE(sigmavirus24): Previously the not_a_symbol check was executed
    # conditionally. Since it is now *always* executed, text may be
    # None. In that case we get a TypeError for `text not in str`.
    not_a_symbol = text and text not in _SYMBOLIC_OPS
    # The % character is strictly speaking a binary operator, but the
    # common usage seems to be to put it next to the format parameters,
    # after a line break.
    return ((is_op_token or is_conjunction) and not_a_symbol)


def _break_around_binary_operators(tokens):
    """Private function to reduce duplication.

    This factors out the shared details between
    :func:`break_before_binary_operator` and
    :func:`break_after_binary_operator`.
    """
    line_break = False
    unary_context = True
    # Previous non-newline token types and text
    previous_token_type = None
    previous_text = None
    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.COMMENT:
            continue
        if ('\n' in text or '\r' in text) and token_type != tokenize.STRING:
            line_break = True
        else:
            yield (token_type, text, previous_token_type, previous_text,
                   line_break, unary_context, start)
            unary_context = text in '([{,;'
            line_break = False
            previous_token_type = token_type
            previous_text = text


@register_check
def break_before_binary_operator(logical_line, tokens):
    r"""
    Avoid breaks before binary operators.

    The preferred place to break around a binary operator is after the
    operator, not before it.

    W503: (width == 0\n + height == 0)
    W503: (width == 0\n and height == 0)
    W503: var = (1\n       & ~2)
    W503: var = (1\n       / -2)
    W503: var = (1\n       + -1\n       + -2)

    Okay: foo(\n    -x)
    Okay: foo(x\n    [])
    Okay: x = '''\n''' + ''
    Okay: foo(x,\n    -y)
    Okay: foo(x,  # comment\n    -y)
    """
    for context in _break_around_binary_operators(tokens):
        (token_type, text, previous_token_type, previous_text,
         line_break, unary_context, start) = context
        if (_is_binary_operator(token_type, text) and line_break and
                not unary_context and
                not _is_binary_operator(previous_token_type,
                                        previous_text)):
            yield start, "W503 line break before binary operator"


@register_check
def break_after_binary_operator(logical_line, tokens):
    r"""
    Avoid breaks after binary operators.

    The preferred place to break around a binary operator is before the
    operator, not after it.

    W504: (width == 0 +\n height == 0)
    W504: (width == 0 and\n height == 0)
    W504: var = (1 &\n       ~2)

    Okay: foo(\n    -x)
    Okay: foo(x\n    [])
    Okay: x = '''\n''' + ''
    Okay: x = '' + '''\n'''
    Okay: foo(x,\n    -y)
    Okay: foo(x,  # comment\n    -y)

    The following should be W504 but unary_context is tricky with these
    Okay: var = (1 /\n       -2)
    Okay: var = (1 +\n       -1 +\n       -2)
    """
    prev_start = None
    for context in _break_around_binary_operators(tokens):
        (token_type, text, previous_token_type, previous_text,
         line_break, unary_context, start) = context
        if (_is_binary_operator(previous_token_type, previous_text) and
                line_break and
                not unary_context and
                not _is_binary_operator(token_type, text)):
            yield prev_start, "W504 line break after binary operator"
        prev_start = start


@register_check
def comparison_to_singleton(logical_line, noqa):
    r"""Comparison to singletons should use "is" or "is not".

    Comparisons to singletons like None should always be done
    with "is" or "is not", never the equality operators.

    Okay: if arg is not None:
    E711: if arg != None:
    E711: if None == arg:
    E712: if arg == True:
    E712: if False == arg:

    Also, beware of writing if x when you really mean if x is not None
    -- e.g. when testing whether a variable or argument that defaults to
    None was set to some other value.  The other value might have a type
    (such as a container) that could be false in a boolean context!
    """
    if noqa:
        return

    for match in COMPARE_SINGLETON_REGEX.finditer(logical_line):
        singleton = match.group(1) or match.group(3)
        same = (match.group(2) == '==')

        msg = "'if cond is %s:'" % (('' if same else 'not ') + singleton)
        if singleton in ('None',):
            code = 'E711'
        else:
            code = 'E712'
            nonzero = ((singleton == 'True' and same) or
                       (singleton == 'False' and not same))
            msg += " or 'if %scond:'" % ('' if nonzero else 'not ')
        yield match.start(2), ("%s comparison to %s should be %s" %
                               (code, singleton, msg))


@register_check
def comparison_negative(logical_line):
    r"""Negative comparison should be done using "not in" and "is not".

    Okay: if x not in y:\n    pass
    Okay: assert (X in Y or X is Z)
    Okay: if not (X in Y):\n    pass
    Okay: zz = x is not y
    E713: Z = not X in Y
    E713: if not X.B in Y:\n    pass
    E714: if not X is Y:\n    pass
    E714: Z = not X.B is Y
    """
    match = COMPARE_NEGATIVE_REGEX.search(logical_line)
    if match:
        pos = match.start(1)
        if match.group(2) == 'in':
            yield pos, "E713 test for membership should be 'not in'"
        else:
            yield pos, "E714 test for object identity should be 'is not'"


@register_check
def comparison_type(logical_line, noqa):
    r"""Object type comparisons should always use isinstance().

    Do not compare types directly.

    Okay: if isinstance(obj, int):
    E721: if type(obj) is type(1):

    When checking if an object is a string, keep in mind that it might
    be a unicode string too! In Python 2.3, str and unicode have a
    common base class, basestring, so you can do:

    Okay: if isinstance(obj, basestring):
    Okay: if type(a1) is type(b1):
    """
    match = COMPARE_TYPE_REGEX.search(logical_line)
    if match and not noqa:
        inst = match.group(1)
        if inst and isidentifier(inst) and inst not in SINGLETONS:
            return  # Allow comparison for types which are not obvious
        yield match.start(), "E721 do not compare types, use 'isinstance()'"


@register_check
def bare_except(logical_line, noqa):
    r"""When catching exceptions, mention specific exceptions when
    possible.

    Okay: except Exception:
    Okay: except BaseException:
    E722: except:
    """
    if noqa:
        return

    match = BLANK_EXCEPT_REGEX.match(logical_line)
    if match:
        yield match.start(), "E722 do not use bare 'except'"


@register_check
def ambiguous_identifier(logical_line, tokens):
    r"""Never use the characters 'l', 'O', or 'I' as variable names.

    In some fonts, these characters are indistinguishable from the
    numerals one and zero. When tempted to use 'l', use 'L' instead.

    Okay: L = 0
    Okay: o = 123
    Okay: i = 42
    E741: l = 0
    E741: O = 123
    E741: I = 42

    Variables can be bound in several other contexts, including class
    and function definitions, 'global' and 'nonlocal' statements,
    exception handlers, and 'with' and 'for' statements.
    In addition, we have a special handling for function parameters.

    Okay: except AttributeError as o:
    Okay: with lock as L:
    Okay: foo(l=12)
    Okay: for a in foo(l=12):
    E741: except AttributeError as O:
    E741: with lock as l:
    E741: global I
    E741: nonlocal l
    E741: def foo(l):
    E741: def foo(l=12):
    E741: l = foo(l=12)
    E741: for l in range(10):
    E742: class I(object):
    E743: def l(x):
    """
    is_func_def = False  # Set to true if 'def' is found
    parameter_parentheses_level = 0
    idents_to_avoid = ('l', 'O', 'I')
    prev_type, prev_text, prev_start, prev_end, __ = tokens[0]
    for token_type, text, start, end, line in tokens[1:]:
        ident = pos = None
        # find function definitions
        if prev_text == 'def':
            is_func_def = True
        # update parameter parentheses level
        if parameter_parentheses_level == 0 and \
                prev_type == tokenize.NAME and \
                token_type == tokenize.OP and text == '(':
            parameter_parentheses_level = 1
        elif parameter_parentheses_level > 0 and \
                token_type == tokenize.OP:
            if text == '(':
                parameter_parentheses_level += 1
            elif text == ')':
                parameter_parentheses_level -= 1
        # identifiers on the lhs of an assignment operator
        if token_type == tokenize.OP and '=' in text and \
                parameter_parentheses_level == 0:
            if prev_text in idents_to_avoid:
                ident = prev_text
                pos = prev_start
        # identifiers bound to values with 'as', 'for',
        # 'global', or 'nonlocal'
        if prev_text in ('as', 'for', 'global', 'nonlocal'):
            if text in idents_to_avoid:
                ident = text
                pos = start
        # function parameter definitions
        if is_func_def:
            if text in idents_to_avoid:
                ident = text
                pos = start
        if prev_text == 'class':
            if text in idents_to_avoid:
                yield start, "E742 ambiguous class definition '%s'" % text
        if prev_text == 'def':
            if text in idents_to_avoid:
                yield start, "E743 ambiguous function definition '%s'" % text
        if ident:
            yield pos, "E741 ambiguous variable name '%s'" % ident
        prev_type = token_type
        prev_text = text
        prev_start = start


@register_check
def python_3000_has_key(logical_line, noqa):
    r"""The {}.has_key() method is removed in Python 3: use the 'in'
    operator.

    Okay: if "alph" in d:\n    print d["alph"]
    W601: assert d.has_key('alph')
    """
    pos = logical_line.find('.has_key(')
    if pos > -1 and not noqa:
        yield pos, "W601 .has_key() is deprecated, use 'in'"


@register_check
def python_3000_raise_comma(logical_line):
    r"""When raising an exception, use "raise ValueError('message')".

    The older form is removed in Python 3.

    Okay: raise DummyError("Message")
    W602: raise DummyError, "Message"
    """
    match = RAISE_COMMA_REGEX.match(logical_line)
    if match and not RERAISE_COMMA_REGEX.match(logical_line):
        yield match.end() - 1, "W602 deprecated form of raising exception"


@register_check
def python_3000_not_equal(logical_line):
    r"""New code should always use != instead of <>.

    The older syntax is removed in Python 3.

    Okay: if a != 'no':
    W603: if a <> 'no':
    """
    pos = logical_line.find('<>')
    if pos > -1:
        yield pos, "W603 '<>' is deprecated, use '!='"


@register_check
def python_3000_backticks(logical_line):
    r"""Use repr() instead of backticks in Python 3.

    Okay: val = repr(1 + 2)
    W604: val = `1 + 2`
    """
    pos = logical_line.find('`')
    if pos > -1:
        yield pos, "W604 backticks are deprecated, use 'repr()'"


@register_check
def python_3000_invalid_escape_sequence(logical_line, tokens, noqa):
    r"""Invalid escape sequences are deprecated in Python 3.6.

    Okay: regex = r'\.png$'
    W605: regex = '\.png$'
    """
    if noqa:
        return

    # https://docs.python.org/3/reference/lexical_analysis.html#string-and-bytes-literals
    valid = [
        '\n',
        '\\',
        '\'',
        '"',
        'a',
        'b',
        'f',
        'n',
        'r',
        't',
        'v',
        '0', '1', '2', '3', '4', '5', '6', '7',
        'x',

        # Escape sequences only recognized in string literals
        'N',
        'u',
        'U',
    ]

    for token_type, text, start, end, line in tokens:
        if token_type == tokenize.STRING:
            start_line, start_col = start
            quote = text[-3:] if text[-3:] in ('"""', "'''") else text[-1]
            # Extract string modifiers (e.g. u or r)
            quote_pos = text.index(quote)
            prefix = text[:quote_pos].lower()
            start = quote_pos + len(quote)
            string = text[start:-len(quote)]

            if 'r' not in prefix:
                pos = string.find('\\')
                while pos >= 0:
                    pos += 1
                    if string[pos] not in valid:
                        line = start_line + string.count('\n', 0, pos)
                        if line == start_line:
                            col = start_col + len(prefix) + len(quote) + pos
                        else:
                            col = pos - string.rfind('\n', 0, pos) - 1
                        yield (
                            (line, col - 1),
                            "W605 invalid escape sequence '\\%s'" %
                            string[pos],
                        )
                    pos = string.find('\\', pos + 1)


@register_check
def python_3000_async_await_keywords(logical_line, tokens):
    """'async' and 'await' are reserved keywords starting at Python 3.7.

    W606: async = 42
    W606: await = 42
    Okay: async def read(db):\n    data = await db.fetch('SELECT ...')
    """
    # The Python tokenize library before Python 3.5 recognizes
    # async/await as a NAME token. Therefore, use a state machine to
    # look for the possible async/await constructs as defined by the
    # Python grammar:
    # https://docs.python.org/3/reference/grammar.html

    state = None
    for token_type, text, start, end, line in tokens:
        error = False

        if token_type == tokenize.NL:
            continue

        if state is None:
            if token_type == tokenize.NAME:
                if text == 'async':
                    state = ('async_stmt', start)
                elif text == 'await':
                    state = ('await', start)
                elif (token_type == tokenize.NAME and
                      text in ('def', 'for')):
                    state = ('define', start)

        elif state[0] == 'async_stmt':
            if token_type == tokenize.NAME and text in ('def', 'with', 'for'):
                # One of funcdef, with_stmt, or for_stmt. Return to
                # looking for async/await names.
                state = None
            else:
                error = True
        elif state[0] == 'await':
            if token_type == tokenize.NAME:
                # An await expression. Return to looking for async/await
                # names.
                state = None
            elif token_type == tokenize.OP and text == '(':
                state = None
            else:
                error = True
        elif state[0] == 'define':
            if token_type == tokenize.NAME and text in ('async', 'await'):
                error = True
            else:
                state = None

        if error:
            yield (
                state[1],
                "W606 'async' and 'await' are reserved keywords starting with "
                "Python 3.7",
            )
            state = None

    # Last token
    if state is not None:
        yield (
            state[1],
            "W606 'async' and 'await' are reserved keywords starting with "
            "Python 3.7",
        )


########################################################################
@register_check
def maximum_doc_length(logical_line, max_doc_length, noqa, tokens):
    r"""Limit all doc lines to a maximum of 72 characters.

    For flowing long blocks of text (docstrings or comments), limiting
    the length to 72 characters is recommended.

    Reports warning W505
    """
    if max_doc_length is None or noqa:
        return

    prev_token = None
    skip_lines = set()
    # Skip lines that
    for token_type, text, start, end, line in tokens:
        if token_type not in SKIP_COMMENTS.union([tokenize.STRING]):
            skip_lines.add(line)

    for token_type, text, start, end, line in tokens:
        # Skip lines that aren't pure strings
        if token_type == tokenize.STRING and skip_lines:
            continue
        if token_type in (tokenize.STRING, tokenize.COMMENT):
            # Only check comment-only lines
            if prev_token is None or prev_token in SKIP_TOKENS:
                lines = line.splitlines()
                for line_num, physical_line in enumerate(lines):
                    if hasattr(physical_line, 'decode'):  # Python 2
                        # The line could contain multi-byte characters
                        try:
                            physical_line = physical_line.decode('utf-8')
                        except UnicodeError:
                            pass
                    if start[0] + line_num == 1 and line.startswith('#!'):
                        return
                    length = len(physical_line)
                    chunks = physical_line.split()
                    if token_type == tokenize.COMMENT:
                        if (len(chunks) == 2 and
                                length - len(chunks[-1]) < MAX_DOC_LENGTH):
                            continue
                    if len(chunks) == 1 and line_num + 1 < len(lines):
                        if (len(chunks) == 1 and
                                length - len(chunks[-1]) < MAX_DOC_LENGTH):
                            continue
                    if length > max_doc_length:
                        doc_error = (start[0] + line_num, max_doc_length)
                        yield (doc_error, "W505 doc line too long "
                                          "(%d > %d characters)"
                               % (length, max_doc_length))
        prev_token = token_type


########################################################################
# Helper functions
########################################################################


if sys.version_info < (3,):
    # Python 2: implicit encoding.
    def readlines(filename):
        """Read the source code."""
        with open(filename, 'rU') as f:
            return f.readlines()
    isidentifier = re.compile(r'[a-zA-Z_]\w*$').match
    stdin_get_value = sys.stdin.read
else:
    # Python 3
    def readlines(filename):
        """Read the source code."""
        try:
            with tokenize.open(filename) as f:
                return f.readlines()
        except (LookupError, SyntaxError, UnicodeError):
            # Fall back if file encoding is improperly declared
            with open(filename, encoding='latin-1') as f:
                return f.readlines()
    isidentifier = str.isidentifier

    def stdin_get_value():
        """Read the value from stdin."""
        return TextIOWrapper(sys.stdin.buffer, errors='ignore').read()

noqa = lru_cache(512)(re.compile(r'# no(?:qa|pep8)\b', re.I).search)


def expand_indent(line):
    r"""Return the amount of indentation.

    Tabs are expanded to the next multiple of 8.

    >>> expand_indent('    ')
    4
    >>> expand_indent('\t')
    8
    >>> expand_indent('       \t')
    8
    >>> expand_indent('        \t')
    16
    """
    line = line.rstrip('\n\r')
    if '\t' not in line:
        return len(line) - len(line.lstrip())
    result = 0
    for char in line:
        if char == '\t':
            result = result // 8 * 8 + 8
        elif char == ' ':
            result += 1
        else:
            break
    return result


def mute_string(text):
    """Replace contents with 'xxx' to prevent syntax matching.

    >>> mute_string('"abc"')
    '"xxx"'
    >>> mute_string("'''abc'''")
    "'''xxx'''"
    >>> mute_string("r'abc'")
    "r'xxx'"
    """
    # String modifiers (e.g. u or r)
    start = text.index(text[-1]) + 1
    end = len(text) - 1
    # Triple quotes
    if text[-3:] in ('"""', "'''"):
        start += 2
        end -= 2
    return text[:start] + 'x' * (end - start) + text[end:]


def parse_udiff(diff, patterns=None, parent='.'):
    """Return a dictionary of matching lines."""
    # For each file of the diff, the entry key is the filename,
    # and the value is a set of row numbers to consider.
    rv = {}
    path = nrows = None
    for line in diff.splitlines():
        if nrows:
            if line[:1] != '-':
                nrows -= 1
            continue
        if line[:3] == '@@ ':
            hunk_match = HUNK_REGEX.match(line)
            (row, nrows) = [int(g or '1') for g in hunk_match.groups()]
            rv[path].update(range(row, row + nrows))
        elif line[:3] == '+++':
            path = line[4:].split('\t', 1)[0]
            # Git diff will use (i)ndex, (w)ork tree, (c)ommit and
            # (o)bject instead of a/b/c/d as prefixes for patches
            if path[:2] in ('b/', 'w/', 'i/'):
                path = path[2:]
            rv[path] = set()
    return {
        os.path.join(parent, filepath): rows
        for (filepath, rows) in rv.items()
        if rows and filename_match(filepath, patterns)
    }


def normalize_paths(value, parent=os.curdir):
    """Parse a comma-separated list of paths.

    Return a list of absolute paths.
    """
    if not value:
        return []
    if isinstance(value, list):
        return value
    paths = []
    for path in value.split(','):
        path = path.strip()
        if '/' in path:
            path = os.path.abspath(os.path.join(parent, path))
        paths.append(path.rstrip('/'))
    return paths


def filename_match(filename, patterns, default=True):
    """Check if patterns contains a pattern that matches filename.

    If patterns is unspecified, this always returns True.
    """
    if not patterns:
        return default
    return any(fnmatch(filename, pattern) for pattern in patterns)


def update_counts(s, counts):
    r"""Adds one to the counts of each appearance of characters in s,
        for characters in counts"""
    for char in s:
        if char in counts:
            counts[char] += 1


def _is_eol_token(token):
    return token[0] in NEWLINE or token[4][token[3][1]:].lstrip() == '\\\n'


########################################################################
# Framework to run all checks
########################################################################


class Checker(object):
    """Load a Python source file, tokenize it, check coding style."""

    def __init__(self, filename=None, lines=None,
                 options=None, report=None, **kwargs):
        if options is None:
            options = StyleGuide(kwargs).options
        else:
            assert not kwargs
        self._io_error = None
        self._physical_checks = options.physical_checks
        self._logical_checks = options.logical_checks
        self._ast_checks = options.ast_checks
        self.max_line_length = options.max_line_length
        self.max_doc_length = options.max_doc_length
        self.indent_size = options.indent_size
        self.multiline = False  # in a multiline string?
        self.hang_closing = options.hang_closing
        self.indent_size = options.indent_size
        self.verbose = options.verbose
        self.filename = filename
        # Dictionary where a checker can store its custom state.
        self._checker_states = {}
        if filename is None:
            self.filename = 'stdin'
            self.lines = lines or []
        elif filename == '-':
            self.filename = 'stdin'
            self.lines = stdin_get_value().splitlines(True)
        elif lines is None:
            try:
                self.lines = readlines(filename)
            except IOError:
                (exc_type, exc) = sys.exc_info()[:2]
                self._io_error = '%s: %s' % (exc_type.__name__, exc)
                self.lines = []
        else:
            self.lines = lines
        if self.lines:
            ord0 = ord(self.lines[0][0])
            if ord0 in (0xef, 0xfeff):  # Strip the UTF-8 BOM
                if ord0 == 0xfeff:
                    self.lines[0] = self.lines[0][1:]
                elif self.lines[0][:3] == '\xef\xbb\xbf':
                    self.lines[0] = self.lines[0][3:]
        self.report = report or options.report
        self.report_error = self.report.error
        self.noqa = False

    def report_invalid_syntax(self):
        """Check if the syntax is valid."""
        (exc_type, exc) = sys.exc_info()[:2]
        if len(exc.args) > 1:
            offset = exc.args[1]
            if len(offset) > 2:
                offset = offset[1:3]
        else:
            offset = (1, 0)
        self.report_error(offset[0], offset[1] or 0,
                          'E901 %s: %s' % (exc_type.__name__, exc.args[0]),
                          self.report_invalid_syntax)

    def readline(self):
        """Get the next line from the input buffer."""
        if self.line_number >= self.total_lines:
            return ''
        line = self.lines[self.line_number]
        self.line_number += 1
        if self.indent_char is None and line[:1] in WHITESPACE:
            self.indent_char = line[0]
        return line

    def run_check(self, check, argument_names):
        """Run a check plugin."""
        arguments = []
        for name in argument_names:
            arguments.append(getattr(self, name))
        return check(*arguments)

    def init_checker_state(self, name, argument_names):
        """Prepare custom state for the specific checker plugin."""
        if 'checker_state' in argument_names:
            self.checker_state = self._checker_states.setdefault(name, {})

    def check_physical(self, line):
        """Run all physical checks on a raw input line."""
        self.physical_line = line
        for name, check, argument_names in self._physical_checks:
            self.init_checker_state(name, argument_names)
            result = self.run_check(check, argument_names)
            if result is not None:
                (offset, text) = result
                self.report_error(self.line_number, offset, text, check)
                if text[:4] == 'E101':
                    self.indent_char = line[0]

    def build_tokens_line(self):
        """Build a logical line from tokens."""
        logical = []
        comments = []
        length = 0
        prev_row = prev_col = mapping = None
        for token_type, text, start, end, line in self.tokens:
            if token_type in SKIP_TOKENS:
                continue
            if not mapping:
                mapping = [(0, start)]
            if token_type == tokenize.COMMENT:
                comments.append(text)
                continue
            if token_type == tokenize.STRING:
                text = mute_string(text)
            if prev_row:
                (start_row, start_col) = start
                if prev_row != start_row:    # different row
                    prev_text = self.lines[prev_row - 1][prev_col - 1]
                    if prev_text == ',' or (prev_text not in '{[(' and
                                            text not in '}])'):
                        text = ' ' + text
                elif prev_col != start_col:  # different column
                    text = line[prev_col:start_col] + text
            logical.append(text)
            length += len(text)
            mapping.append((length, end))
            (prev_row, prev_col) = end
        self.logical_line = ''.join(logical)
        self.noqa = comments and noqa(''.join(comments))
        return mapping

    def check_logical(self):
        """Build a line from tokens and run all logical checks on it."""
        self.report.increment_logical_line()
        mapping = self.build_tokens_line()
        if not mapping:
            return

        mapping_offsets = [offset for offset, _ in mapping]
        (start_row, start_col) = mapping[0][1]
        start_line = self.lines[start_row - 1]
        self.indent_level = expand_indent(start_line[:start_col])
        if self.blank_before < self.blank_lines:
            self.blank_before = self.blank_lines
        if self.verbose >= 2:
            print(self.logical_line[:80].rstrip())
        for name, check, argument_names in self._logical_checks:
            if self.verbose >= 4:
                print('   ' + name)
            self.init_checker_state(name, argument_names)
            for offset, text in self.run_check(check, argument_names) or ():
                if not isinstance(offset, tuple):
                    # As mappings are ordered, bisecting is a fast way
                    # to find a given offset in them.
                    token_offset, pos = mapping[bisect.bisect_left(
                        mapping_offsets, offset)]
                    offset = (pos[0], pos[1] + offset - token_offset)
                self.report_error(offset[0], offset[1], text, check)
        if self.logical_line:
            self.previous_indent_level = self.indent_level
            self.previous_logical = self.logical_line
            if not self.indent_level:
                self.previous_unindented_logical_line = self.logical_line
        self.blank_lines = 0
        self.tokens = []

    def check_ast(self):
        """Build the file's AST and run all AST checks."""
        try:
            tree = compile(''.join(self.lines), '', 'exec', PyCF_ONLY_AST)
        except (ValueError, SyntaxError, TypeError):
            return self.report_invalid_syntax()
        for name, cls, __ in self._ast_checks:
            checker = cls(tree, self.filename)
            for lineno, offset, text, check in checker.run():
                if not self.lines or not noqa(self.lines[lineno - 1]):
                    self.report_error(lineno, offset, text, check)

    def generate_tokens(self):
        """Tokenize file, run physical line checks and yield tokens."""
        if self._io_error:
            self.report_error(1, 0, 'E902 %s' % self._io_error, readlines)
        tokengen = tokenize.generate_tokens(self.readline)
        try:
            prev_physical = ''
            for token in tokengen:
                if token[2][0] > self.total_lines:
                    return
                self.noqa = token[4] and noqa(token[4])
                self.maybe_check_physical(token, prev_physical)
                yield token
                prev_physical = token[4]
        except (SyntaxError, tokenize.TokenError):
            self.report_invalid_syntax()

    def maybe_check_physical(self, token, prev_physical):
        """If appropriate for token, check current physical line(s)."""
        # Called after every token, but act only on end of line.

        # a newline token ends a single physical line.
        if _is_eol_token(token):
            # if the file does not end with a newline, the NEWLINE
            # token is inserted by the parser, but it does not contain
            # the previous physical line in `token[4]`
            if token[4] == '':
                self.check_physical(prev_physical)
            else:
                self.check_physical(token[4])
        elif token[0] == tokenize.STRING and '\n' in token[1]:
            # Less obviously, a string that contains newlines is a
            # multiline string, either triple-quoted or with internal
            # newlines backslash-escaped. Check every physical line in
            # the string *except* for the last one: its newline is
            # outside of the multiline string, so we consider it a
            # regular physical line, and will check it like any other
            # physical line.
            #
            # Subtleties:
            # - we don't *completely* ignore the last line; if it
            #   contains the magical "# noqa" comment, we disable all
            #   physical checks for the entire multiline string
            # - have to wind self.line_number back because initially it
            #   points to the last line of the string, and we want
            #   check_physical() to give accurate feedback
            if noqa(token[4]):
                return
            self.multiline = True
            self.line_number = token[2][0]
            _, src, (_, offset), _, _ = token
            src = self.lines[self.line_number - 1][:offset] + src
            for line in src.split('\n')[:-1]:
                self.check_physical(line + '\n')
                self.line_number += 1
            self.multiline = False

    def check_all(self, expected=None, line_offset=0):
        """Run all checks on the input file."""
        self.report.init_file(self.filename, self.lines, expected, line_offset)
        self.total_lines = len(self.lines)
        if self._ast_checks:
            self.check_ast()
        self.line_number = 0
        self.indent_char = None
        self.indent_level = self.previous_indent_level = 0
        self.previous_logical = ''
        self.previous_unindented_logical_line = ''
        self.tokens = []
        self.blank_lines = self.blank_before = 0
        parens = 0
        for token in self.generate_tokens():
            self.tokens.append(token)
            token_type, text = token[0:2]
            if self.verbose >= 3:
                if token[2][0] == token[3][0]:
                    pos = '[%s:%s]' % (token[2][1] or '', token[3][1])
                else:
                    pos = 'l.%s' % token[3][0]
                print('l.%s\t%s\t%s\t%r' %
                      (token[2][0], pos, tokenize.tok_name[token[0]], text))
            if token_type == tokenize.OP:
                if text in '([{':
                    parens += 1
                elif text in '}])':
                    parens -= 1
            elif not parens:
                if token_type in NEWLINE:
                    if token_type == tokenize.NEWLINE:
                        self.check_logical()
                        self.blank_before = 0
                    elif len(self.tokens) == 1:
                        # The physical line contains only this token.
                        self.blank_lines += 1
                        del self.tokens[0]
                    else:
                        self.check_logical()
        if self.tokens:
            self.check_physical(self.lines[-1])
            self.check_logical()
        return self.report.get_file_results()


class BaseReport(object):
    """Collect the results of the checks."""

    print_filename = False

    def __init__(self, options):
        self._benchmark_keys = options.benchmark_keys
        self._ignore_code = options.ignore_code
        # Results
        self.elapsed = 0
        self.total_errors = 0
        self.counters = dict.fromkeys(self._benchmark_keys, 0)
        self.messages = {}

    def start(self):
        """Start the timer."""
        self._start_time = time.time()

    def stop(self):
        """Stop the timer."""
        self.elapsed = time.time() - self._start_time

    def init_file(self, filename, lines, expected, line_offset):
        """Signal a new file."""
        self.filename = filename
        self.lines = lines
        self.expected = expected or ()
        self.line_offset = line_offset
        self.file_errors = 0
        self.counters['files'] += 1
        self.counters['physical lines'] += len(lines)

    def increment_logical_line(self):
        """Signal a new logical line."""
        self.counters['logical lines'] += 1

    def error(self, line_number, offset, text, check):
        """Report an error, according to options."""
        code = text[:4]
        if self._ignore_code(code):
            return
        if code in self.counters:
            self.counters[code] += 1
        else:
            self.counters[code] = 1
            self.messages[code] = text[5:]
        # Don't care about expected errors or warnings
        if code in self.expected:
            return
        if self.print_filename and not self.file_errors:
            print(self.filename)
        self.file_errors += 1
        self.total_errors += 1
        return code

    def get_file_results(self):
        """Return the count of errors and warnings for this file."""
        return self.file_errors

    def get_count(self, prefix=''):
        """Return the total count of errors and warnings."""
        return sum(self.counters[key]
                   for key in self.messages if key.startswith(prefix))

    def get_statistics(self, prefix=''):
        """Get statistics for message codes that start with the prefix.

        prefix='' matches all errors and warnings
        prefix='E' matches all errors
        prefix='W' matches all warnings
        prefix='E4' matches all errors that have to do with imports
        """
        return ['%-7s %s %s' % (self.counters[key], key, self.messages[key])
                for key in sorted(self.messages) if key.startswith(prefix)]

    def print_statistics(self, prefix=''):
        """Print overall statistics (number of errors and warnings)."""
        for line in self.get_statistics(prefix):
            print(line)

    def print_benchmark(self):
        """Print benchmark numbers."""
        print('%-7.2f %s' % (self.elapsed, 'seconds elapsed'))
        if self.elapsed:
            for key in self._benchmark_keys:
                print('%-7d %s per second (%d total)' %
                      (self.counters[key] / self.elapsed, key,
                       self.counters[key]))


class FileReport(BaseReport):
    """Collect the results of the checks and print the filenames."""

    print_filename = True


class StandardReport(BaseReport):
    """Collect and print the results of the checks."""

    def __init__(self, options):
        super(StandardReport, self).__init__(options)
        self._fmt = REPORT_FORMAT.get(options.format.lower(),
                                      options.format)
        self._repeat = options.repeat
        self._show_source = options.show_source
        self._show_pep8 = options.show_pep8

    def init_file(self, filename, lines, expected, line_offset):
        """Signal a new file."""
        self._deferred_print = []
        return super(StandardReport, self).init_file(
            filename, lines, expected, line_offset)

    def error(self, line_number, offset, text, check):
        """Report an error, according to options."""
        code = super(StandardReport, self).error(line_number, offset,
                                                 text, check)
        if code and (self.counters[code] == 1 or self._repeat):
            self._deferred_print.append(
                (line_number, offset, code, text[5:], check.__doc__))
        return code

    def get_file_results(self):
        """Print results and return the overall count for this file."""
        self._deferred_print.sort()
        for line_number, offset, code, text, doc in self._deferred_print:
            print(self._fmt % {
                'path': self.filename,
                'row': self.line_offset + line_number, 'col': offset + 1,
                'code': code, 'text': text,
            })
            if self._show_source:
                if line_number > len(self.lines):
                    line = ''
                else:
                    line = self.lines[line_number - 1]
                print(line.rstrip())
                print(re.sub(r'\S', ' ', line[:offset]) + '^')
            if self._show_pep8 and doc:
                print('    ' + doc.strip())

            # stdout is block buffered when not stdout.isatty().
            # line can be broken where buffer boundary since other
            # processes write to same file.
            # flush() after print() to avoid buffer boundary.
            # Typical buffer size is 8192. line written safely when
            # len(line) < 8192.
            sys.stdout.flush()
        return self.file_errors


class DiffReport(StandardReport):
    """Collect and print the results for the changed lines only."""

    def __init__(self, options):
        super(DiffReport, self).__init__(options)
        self._selected = options.selected_lines

    def error(self, line_number, offset, text, check):
        if line_number not in self._selected[self.filename]:
            return
        return super(DiffReport, self).error(line_number, offset, text, check)


class StyleGuide(object):
    """Initialize a PEP-8 instance with few options."""

    def __init__(self, *args, **kwargs):
        # build options from the command line
        self.checker_class = kwargs.pop('checker_class', Checker)
        parse_argv = kwargs.pop('parse_argv', False)
        config_file = kwargs.pop('config_file', False)
        parser = kwargs.pop('parser', None)
        # build options from dict
        options_dict = dict(*args, **kwargs)
        arglist = None if parse_argv else options_dict.get('paths', None)
        verbose = options_dict.get('verbose', None)
        options, self.paths = process_options(
            arglist, parse_argv, config_file, parser, verbose)
        if options_dict:
            options.__dict__.update(options_dict)
            if 'paths' in options_dict:
                self.paths = options_dict['paths']

        self.runner = self.input_file
        self.options = options

        if not options.reporter:
            options.reporter = BaseReport if options.quiet else StandardReport

        options.select = tuple(options.select or ())
        if not (options.select or options.ignore or
                options.testsuite or options.doctest) and DEFAULT_IGNORE:
            # The default choice: ignore controversial checks
            options.ignore = tuple(DEFAULT_IGNORE.split(','))
        else:
            # Ignore all checks which are not explicitly selected
            options.ignore = ('',) if options.select else tuple(options.ignore)
        options.benchmark_keys = BENCHMARK_KEYS[:]
        options.ignore_code = self.ignore_code
        options.physical_checks = self.get_checks('physical_line')
        options.logical_checks = self.get_checks('logical_line')
        options.ast_checks = self.get_checks('tree')
        self.init_report()

    def init_report(self, reporter=None):
        """Initialize the report instance."""
        self.options.report = (reporter or self.options.reporter)(self.options)
        return self.options.report

    def check_files(self, paths=None):
        """Run all checks on the paths."""
        if paths is None:
            paths = self.paths
        report = self.options.report
        runner = self.runner
        report.start()
        try:
            for path in paths:
                if os.path.isdir(path):
                    self.input_dir(path)
                elif not self.excluded(path):
                    runner(path)
        except KeyboardInterrupt:
            print('... stopped')
        report.stop()
        return report

    def input_file(self, filename, lines=None, expected=None, line_offset=0):
        """Run all checks on a Python source file."""
        if self.options.verbose:
            print('checking %s' % filename)
        fchecker = self.checker_class(
            filename, lines=lines, options=self.options)
        return fchecker.check_all(expected=expected, line_offset=line_offset)

    def input_dir(self, dirname):
        """Check all files in this directory and all subdirectories."""
        dirname = dirname.rstrip('/')
        if self.excluded(dirname):
            return 0
        counters = self.options.report.counters
        verbose = self.options.verbose
        filepatterns = self.options.filename
        runner = self.runner
        for root, dirs, files in os.walk(dirname):
            if verbose:
                print('directory ' + root)
            counters['directories'] += 1
            for subdir in sorted(dirs):
                if self.excluded(subdir, root):
                    dirs.remove(subdir)
            for filename in sorted(files):
                # contain a pattern that matches?
                if ((filename_match(filename, filepatterns) and
                     not self.excluded(filename, root))):
                    runner(os.path.join(root, filename))

    def excluded(self, filename, parent=None):
        """Check if the file should be excluded.

        Check if 'options.exclude' contains a pattern matching filename.
        """
        if not self.options.exclude:
            return False
        basename = os.path.basename(filename)
        if filename_match(basename, self.options.exclude):
            return True
        if parent:
            filename = os.path.join(parent, filename)
        filename = os.path.abspath(filename)
        return filename_match(filename, self.options.exclude)

    def ignore_code(self, code):
        """Check if the error code should be ignored.

        If 'options.select' contains a prefix of the error code,
        return False.  Else, if 'options.ignore' contains a prefix of
        the error code, return True.
        """
        if len(code) < 4 and any(s.startswith(code)
                                 for s in self.options.select):
            return False
        return (code.startswith(self.options.ignore) and
                not code.startswith(self.options.select))

    def get_checks(self, argument_name):
        """Get all the checks for this category.

        Find all globally visible functions where the first argument
        name starts with argument_name and which contain selected tests.
        """
        checks = []
        for check, attrs in _checks[argument_name].items():
            (codes, args) = attrs
            if any(not (code and self.ignore_code(code)) for code in codes):
                checks.append((check.__name__, check, args))
        return sorted(checks)


def get_parser(prog='pycodestyle', version=__version__):
    """Create the parser for the program."""
    parser = OptionParser(prog=prog, version=version,
                          usage="%prog [options] input ...")
    parser.config_options = [
        'exclude', 'filename', 'select', 'ignore', 'max-line-length',
        'max-doc-length', 'indent-size', 'hang-closing', 'count', 'format',
        'quiet', 'show-pep8', 'show-source', 'statistics', 'verbose']
    parser.add_option('-v', '--verbose', default=0, action='count',
                      help="print status messages, or debug with -vv")
    parser.add_option('-q', '--quiet', default=0, action='count',
                      help="report only file names, or nothing with -qq")
    parser.add_option('-r', '--repeat', default=True, action='store_true',
                      help="(obsolete) show all occurrences of the same error")
    parser.add_option('--first', action='store_false', dest='repeat',
                      help="show first occurrence of each error")
    parser.add_option('--exclude', metavar='patterns', default=DEFAULT_EXCLUDE,
                      help="exclude files or directories which match these "
                           "comma separated patterns (default: %default)")
    parser.add_option('--filename', metavar='patterns', default='*.py',
                      help="when parsing directories, only check filenames "
                           "matching these comma separated patterns "
                           "(default: %default)")
    parser.add_option('--select', metavar='errors', default='',
                      help="select errors and warnings (e.g. E,W6)")
    parser.add_option('--ignore', metavar='errors', default='',
                      help="skip errors and warnings (e.g. E4,W) "
                           "(default: %s)" % DEFAULT_IGNORE)
    parser.add_option('--show-source', action='store_true',
                      help="show source code for each error")
    parser.add_option('--show-pep8', action='store_true',
                      help="show text of PEP 8 for each error "
                           "(implies --first)")
    parser.add_option('--statistics', action='store_true',
                      help="count errors and warnings")
    parser.add_option('--count', action='store_true',
                      help="print total number of errors and warnings "
                           "to standard error and set exit code to 1 if "
                           "total is not null")
    parser.add_option('--max-line-length', type='int', metavar='n',
                      default=MAX_LINE_LENGTH,
                      help="set maximum allowed line length "
                           "(default: %default)")
    parser.add_option('--max-doc-length', type='int', metavar='n',
                      default=None,
                      help="set maximum allowed doc line length and perform "
                           "these checks (unchecked if not set)")
    parser.add_option('--indent-size', type='int', metavar='n',
                      default=INDENT_SIZE,
                      help="set how many spaces make up an indent "
                           "(default: %default)")
    parser.add_option('--hang-closing', action='store_true',
                      help="hang closing bracket instead of matching "
                           "indentation of opening bracket's line")
    parser.add_option('--format', metavar='format', default='default',
                      help="set the error format [default|pylint|<custom>]")
    parser.add_option('--diff', action='store_true',
                      help="report changes only within line number ranges in "
                           "the unified diff received on STDIN")
    group = parser.add_option_group("Testing Options")
    if os.path.exists(TESTSUITE_PATH):
        group.add_option('--testsuite', metavar='dir',
                         help="run regression tests from dir")
        group.add_option('--doctest', action='store_true',
                         help="run doctest on myself")
    group.add_option('--benchmark', action='store_true',
                     help="measure processing speed")
    return parser


def read_config(options, args, arglist, parser):
    """Read and parse configurations.

    If a config file is specified on the command line with the
    "--config" option, then only it is used for configuration.

    Otherwise, the user configuration (~/.config/pycodestyle) and any
    local configurations in the current directory or above will be
    merged together (in that order) using the read method of
    ConfigParser.
    """
    config = RawConfigParser()

    cli_conf = options.config

    local_dir = os.curdir

    if USER_CONFIG and os.path.isfile(USER_CONFIG):
        if options.verbose:
            print('user configuration: %s' % USER_CONFIG)
        config.read(USER_CONFIG)

    parent = tail = args and os.path.abspath(os.path.commonprefix(args))
    while tail:
        if config.read(os.path.join(parent, fn) for fn in PROJECT_CONFIG):
            local_dir = parent
            if options.verbose:
                print('local configuration: in %s' % parent)
            break
        (parent, tail) = os.path.split(parent)

    if cli_conf and os.path.isfile(cli_conf):
        if options.verbose:
            print('cli configuration: %s' % cli_conf)
        config.read(cli_conf)

    pycodestyle_section = None
    if config.has_section(parser.prog):
        pycodestyle_section = parser.prog
    elif config.has_section('pep8'):
        pycodestyle_section = 'pep8'  # Deprecated
        warnings.warn('[pep8] section is deprecated. Use [pycodestyle].')

    if pycodestyle_section:
        option_list = {o.dest: o.type or o.action for o in parser.option_list}

        # First, read the default values
        (new_options, __) = parser.parse_args([])

        # Second, parse the configuration
        for opt in config.options(pycodestyle_section):
            if opt.replace('_', '-') not in parser.config_options:
                print("  unknown option '%s' ignored" % opt)
                continue
            if options.verbose > 1:
                print("  %s = %s" % (opt,
                                     config.get(pycodestyle_section, opt)))
            normalized_opt = opt.replace('-', '_')
            opt_type = option_list[normalized_opt]
            if opt_type in ('int', 'count'):
                value = config.getint(pycodestyle_section, opt)
            elif opt_type in ('store_true', 'store_false'):
                value = config.getboolean(pycodestyle_section, opt)
            else:
                value = config.get(pycodestyle_section, opt)
                if normalized_opt == 'exclude':
                    value = normalize_paths(value, local_dir)
            setattr(new_options, normalized_opt, value)

        # Third, overwrite with the command-line options
        (options, __) = parser.parse_args(arglist, values=new_options)
    options.doctest = options.testsuite = False
    return options


def process_options(arglist=None, parse_argv=False, config_file=None,
                    parser=None, verbose=None):
    """Process options passed either via arglist or command line args.

    Passing in the ``config_file`` parameter allows other tools, such as
    flake8 to specify their own options to be processed in pycodestyle.
    """
    if not parser:
        parser = get_parser()
    if not parser.has_option('--config'):
        group = parser.add_option_group("Configuration", description=(
            "The project options are read from the [%s] section of the "
            "tox.ini file or the setup.cfg file located in any parent folder "
            "of the path(s) being processed.  Allowed options are: %s." %
            (parser.prog, ', '.join(parser.config_options))))
        group.add_option('--config', metavar='path', default=config_file,
                         help="user config file location")
    # Don't read the command line if the module is used as a library.
    if not arglist and not parse_argv:
        arglist = []
    # If parse_argv is True and arglist is None, arguments are
    # parsed from the command line (sys.argv)
    (options, args) = parser.parse_args(arglist)
    options.reporter = None

    # If explicitly specified verbosity, override any `-v` CLI flag
    if verbose is not None:
        options.verbose = verbose

    if options.ensure_value('testsuite', False):
        args.append(options.testsuite)
    elif not options.ensure_value('doctest', False):
        if parse_argv and not args:
            if options.diff or any(os.path.exists(name)
                                   for name in PROJECT_CONFIG):
                args = ['.']
            else:
                parser.error('input not specified')
        options = read_config(options, args, arglist, parser)
        options.reporter = parse_argv and options.quiet == 1 and FileReport

    options.filename = _parse_multi_options(options.filename)
    options.exclude = normalize_paths(options.exclude)
    options.select = _parse_multi_options(options.select)
    options.ignore = _parse_multi_options(options.ignore)

    if options.diff:
        options.reporter = DiffReport
        stdin = stdin_get_value()
        options.selected_lines = parse_udiff(stdin, options.filename, args[0])
        args = sorted(options.selected_lines)

    return options, args


def _parse_multi_options(options, split_token=','):
    r"""Split and strip and discard empties.

    Turns the following:

    A,
    B,

    into ["A", "B"]
    """
    if options:
        return [o.strip() for o in options.split(split_token) if o.strip()]
    else:
        return options


def _main():
    """Parse options and run checks on Python source."""
    import signal

    # Handle "Broken pipe" gracefully
    try:
        signal.signal(signal.SIGPIPE, lambda signum, frame: sys.exit(1))
    except AttributeError:
        pass    # not supported on Windows

    style_guide = StyleGuide(parse_argv=True)
    options = style_guide.options

    if options.doctest or options.testsuite:
        from testsuite.support import run_tests
        report = run_tests(style_guide)
    else:
        report = style_guide.check_files()

    if options.statistics:
        report.print_statistics()

    if options.benchmark:
        report.print_benchmark()

    if options.testsuite and not options.quiet:
        report.print_results()

    if report.total_errors:
        if options.count:
            sys.stderr.write(str(report.total_errors) + '\n')
        sys.exit(1)


if __name__ == '__main__':
    _main()
