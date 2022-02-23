"""
Unit tests for the stack formatting utilities
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

import imp
import os
import re
import sys
import pytest

from joblib.format_stack import safe_repr, _fixed_getframes, format_records
from joblib.format_stack import format_exc
from joblib.test.common import with_numpy, np

###############################################################################


class Vicious(object):
    def __repr__(self):
        raise ValueError


def test_safe_repr():
    safe_repr(Vicious())


def _change_file_extensions_to_pyc(record):
    _1, filename, _2, _3, _4, _5 = record
    if filename.endswith('.py'):
        filename += 'c'
    return _1, filename, _2, _3, _4, _5


def _raise_exception(a, b):
    """Function that raises with a non trivial call stack
    """
    def helper(a, b):
        raise ValueError('Nope, this can not work')

    helper(a, b)


def test_format_records():
    try:
        _raise_exception('a', 42)
    except ValueError:
        etb = sys.exc_info()[2]
        records = _fixed_getframes(etb)

        # Modify filenames in traceback records from .py to .pyc
        pyc_records = [_change_file_extensions_to_pyc(record)
                       for record in records]

        formatted_records = format_records(pyc_records)

        # Check that the .py file and not the .pyc one is listed in
        # the traceback
        for fmt_rec in formatted_records:
            assert 'test_format_stack.py in' in fmt_rec

        # Check exception stack
        arrow_regex = r'^-+>\s+\d+\s+'
        assert re.search(arrow_regex + r"_raise_exception\('a', 42\)",
                         formatted_records[0],
                         re.MULTILINE)
        assert re.search(arrow_regex + r'helper\(a, b\)',
                         formatted_records[1],
                         re.MULTILINE)
        assert "a = 'a'" in formatted_records[1]
        assert 'b = 42' in formatted_records[1]
        assert re.search(arrow_regex +
                         r"raise ValueError\('Nope, this can not work'\)",
                         formatted_records[2],
                         re.MULTILINE)


def test_format_records_file_with_less_lines_than_context(tmpdir):
    # See https://github.com/joblib/joblib/issues/420
    filename = os.path.join(tmpdir.strpath, 'small_file.py')
    code_lines = ['def func():', '    1/0']
    code = '\n'.join(code_lines)
    with open(filename, 'w') as f:
        f.write(code)

    small_file = imp.load_source('small_file', filename)
    if not hasattr(small_file, 'func'):
        pytest.skip("PyPy bug?")
    try:
        small_file.func()
    except ZeroDivisionError:
        etb = sys.exc_info()[2]

        records = _fixed_getframes(etb, context=10)
        # Check that if context is bigger than the number of lines in
        # the file you do not get padding
        frame, tb_filename, line, func_name, context, _ = records[-1]
        assert [l.rstrip() for l in context] == code_lines

        formatted_records = format_records(records)
        # 2 lines for header in the traceback: lines of ...... +
        # filename with function
        len_header = 2
        nb_lines_formatted_records = len(formatted_records[1].splitlines())
        assert (nb_lines_formatted_records == len_header + len(code_lines))
        # Check exception stack
        arrow_regex = r'^-+>\s+\d+\s+'
        assert re.search(arrow_regex + r'1/0',
                         formatted_records[1],
                         re.MULTILINE)


@with_numpy
def test_format_exc_with_compiled_code():
    # Trying to tokenize compiled C code raise SyntaxError.
    # See https://github.com/joblib/joblib/issues/101 for more details.
    try:
        np.random.uniform('invalid_value')
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        formatted_exc = format_exc(exc_type, exc_value,
                                   exc_traceback, context=10)
        # The name of the extension can be something like
        # mtrand.cpython-33m.so
        pattern = r'mtrand[a-z0-9._-]*\.(so|pyd)'
        assert re.search(pattern, formatted_exc)
