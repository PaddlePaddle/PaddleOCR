"""
Test the logger module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import re

from joblib.logger import PrintTime


def test_print_time(tmpdir, capsys):
    # A simple smoke test for PrintTime.
    logfile = tmpdir.join('test.log').strpath
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    # Create a second time, to smoke test log rotation.
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    # And a third time
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')

    out_printed_text, err_printed_text = capsys.readouterr()
    # Use regexps to be robust to time variations
    match = r"Foo: 0\..s, 0\..min\nFoo: 0\..s, 0..min\nFoo: " + \
            r".\..s, 0..min\n"
    if not re.match(match, err_printed_text):
        raise AssertionError('Excepted %s, got %s' %
                             (match, err_printed_text))
