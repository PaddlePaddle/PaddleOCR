from __future__ import print_function, absolute_import
import os
import tempfile
import unittest
import sys
import re
import warnings
import io
from textwrap import dedent

from future.utils import bind_method, PY26, PY3, PY2, PY27
from future.moves.subprocess import check_output, STDOUT, CalledProcessError

if PY26:
    import unittest2 as unittest


def reformat_code(code):
    """
    Removes any leading \n and dedents.
    """
    if code.startswith('\n'):
        code = code[1:]
    return dedent(code)


def order_future_lines(code):
    """
    Returns the code block with any ``__future__`` import lines sorted, and
    then any ``future`` import lines sorted, then any ``builtins`` import lines
    sorted.

    This only sorts the lines within the expected blocks.

    See test_order_future_lines() for an example.
    """

    # We need .splitlines(keepends=True), which doesn't exist on Py2,
    # so we use this instead:
    lines = code.split('\n')

    uufuture_line_numbers = [i for i, line in enumerate(lines)
                               if line.startswith('from __future__ import ')]

    future_line_numbers = [i for i, line in enumerate(lines)
                             if line.startswith('from future')
                             or line.startswith('from past')]

    builtins_line_numbers = [i for i, line in enumerate(lines)
                             if line.startswith('from builtins')]

    assert code.lstrip() == code, ('internal usage error: '
            'dedent the code before calling order_future_lines()')

    def mymax(numbers):
        return max(numbers) if len(numbers) > 0 else 0

    def mymin(numbers):
        return min(numbers) if len(numbers) > 0 else float('inf')

    assert mymax(uufuture_line_numbers) <= mymin(future_line_numbers), \
            'the __future__ and future imports are out of order'

    # assert mymax(future_line_numbers) <= mymin(builtins_line_numbers), \
    #         'the future and builtins imports are out of order'

    uul = sorted([lines[i] for i in uufuture_line_numbers])
    sorted_uufuture_lines = dict(zip(uufuture_line_numbers, uul))

    fl = sorted([lines[i] for i in future_line_numbers])
    sorted_future_lines = dict(zip(future_line_numbers, fl))

    bl = sorted([lines[i] for i in builtins_line_numbers])
    sorted_builtins_lines = dict(zip(builtins_line_numbers, bl))

    # Replace the old unsorted "from __future__ import ..." lines with the
    # new sorted ones:
    new_lines = []
    for i in range(len(lines)):
        if i in uufuture_line_numbers:
            new_lines.append(sorted_uufuture_lines[i])
        elif i in future_line_numbers:
            new_lines.append(sorted_future_lines[i])
        elif i in builtins_line_numbers:
            new_lines.append(sorted_builtins_lines[i])
        else:
            new_lines.append(lines[i])
    return '\n'.join(new_lines)


class VerboseCalledProcessError(CalledProcessError):
    """
    Like CalledProcessError, but it displays more information (message and
    script output) for diagnosing test failures etc.
    """
    def __init__(self, msg, returncode, cmd, output=None):
        self.msg = msg
        self.returncode = returncode
        self.cmd = cmd
        self.output = output

    def __str__(self):
        return ("Command '%s' failed with exit status %d\nMessage: %s\nOutput: %s"
                % (self.cmd, self.returncode, self.msg, self.output))

class FuturizeError(VerboseCalledProcessError):
    pass

class PasteurizeError(VerboseCalledProcessError):
    pass


class CodeHandler(unittest.TestCase):
    """
    Handy mixin for test classes for writing / reading / futurizing /
    running .py files in the test suite.
    """
    def setUp(self):
        """
        The outputs from the various futurize stages should have the
        following headers:
        """
        # After stage1:
        # TODO: use this form after implementing a fixer to consolidate
        #       __future__ imports into a single line:
        # self.headers1 = """
        # from __future__ import absolute_import, division, print_function
        # """
        self.headers1 = reformat_code("""
        from __future__ import absolute_import
        from __future__ import division
        from __future__ import print_function
        """)

        # After stage2 --all-imports:
        # TODO: use this form after implementing a fixer to consolidate
        #       __future__ imports into a single line:
        # self.headers2 = """
        # from __future__ import (absolute_import, division,
        #                         print_function, unicode_literals)
        # from future import standard_library
        # from future.builtins import *
        # """
        self.headers2 = reformat_code("""
        from __future__ import absolute_import
        from __future__ import division
        from __future__ import print_function
        from __future__ import unicode_literals
        from future import standard_library
        standard_library.install_aliases()
        from builtins import *
        """)
        self.interpreters = [sys.executable]
        self.tempdir = tempfile.mkdtemp() + os.path.sep
        pypath = os.getenv('PYTHONPATH')
        if pypath:
            self.env = {'PYTHONPATH': os.getcwd() + os.pathsep + pypath}
        else:
            self.env = {'PYTHONPATH': os.getcwd()}

    def convert(self, code, stages=(1, 2), all_imports=False, from3=False,
                reformat=True, run=True, conservative=False):
        """
        Converts the code block using ``futurize`` and returns the
        resulting code.

        Passing stages=[1] or stages=[2] passes the flag ``--stage1`` or
        ``stage2`` to ``futurize``. Passing both stages runs ``futurize``
        with both stages by default.

        If from3 is False, runs ``futurize``, converting from Python 2 to
        both 2 and 3. If from3 is True, runs ``pasteurize`` to convert
        from Python 3 to both 2 and 3.

        Optionally reformats the code block first using the reformat() function.

        If run is True, runs the resulting code under all Python
        interpreters in self.interpreters.
        """
        if reformat:
            code = reformat_code(code)
        self._write_test_script(code)
        self._futurize_test_script(stages=stages, all_imports=all_imports,
                                   from3=from3, conservative=conservative)
        output = self._read_test_script()
        if run:
            for interpreter in self.interpreters:
                _ = self._run_test_script(interpreter=interpreter)
        return output

    def compare(self, output, expected, ignore_imports=True):
        """
        Compares whether the code blocks are equal. If not, raises an
        exception so the test fails. Ignores any trailing whitespace like
        blank lines.

        If ignore_imports is True, passes the code blocks into the
        strip_future_imports method.

        If one code block is a unicode string and the other a
        byte-string, it assumes the byte-string is encoded as utf-8.
        """
        if ignore_imports:
            output = self.strip_future_imports(output)
            expected = self.strip_future_imports(expected)
        if isinstance(output, bytes) and not isinstance(expected, bytes):
            output = output.decode('utf-8')
        if isinstance(expected, bytes) and not isinstance(output, bytes):
            expected = expected.decode('utf-8')
        self.assertEqual(order_future_lines(output.rstrip()),
                         expected.rstrip())

    def strip_future_imports(self, code):
        """
        Strips any of these import lines:

            from __future__ import <anything>
            from future <anything>
            from future.<anything>
            from builtins <anything>

        or any line containing:
            install_hooks()
        or:
            install_aliases()

        Limitation: doesn't handle imports split across multiple lines like
        this:

            from __future__ import (absolute_import, division, print_function,
                                    unicode_literals)
        """
        output = []
        # We need .splitlines(keepends=True), which doesn't exist on Py2,
        # so we use this instead:
        for line in code.split('\n'):
            if not (line.startswith('from __future__ import ')
                    or line.startswith('from future ')
                    or line.startswith('from builtins ')
                    or 'install_hooks()' in line
                    or 'install_aliases()' in line
                    # but don't match "from future_builtins" :)
                    or line.startswith('from future.')):
                output.append(line)
        return '\n'.join(output)

    def convert_check(self, before, expected, stages=(1, 2), all_imports=False,
                      ignore_imports=True, from3=False, run=True,
                      conservative=False):
        """
        Convenience method that calls convert() and compare().

        Reformats the code blocks automatically using the reformat_code()
        function.

        If all_imports is passed, we add the appropriate import headers
        for the stage(s) selected to the ``expected`` code-block, so they
        needn't appear repeatedly in the test code.

        If ignore_imports is True, ignores the presence of any lines
        beginning:

            from __future__ import ...
            from future import ...

        for the purpose of the comparison.
        """
        output = self.convert(before, stages=stages, all_imports=all_imports,
                              from3=from3, run=run, conservative=conservative)
        if all_imports:
            headers = self.headers2 if 2 in stages else self.headers1
        else:
            headers = ''

        reformatted = reformat_code(expected)
        if headers in reformatted:
            headers = ''

        self.compare(output, headers + reformatted,
                     ignore_imports=ignore_imports)

    def unchanged(self, code, **kwargs):
        """
        Convenience method to ensure the code is unchanged by the
        futurize process.
        """
        self.convert_check(code, code, **kwargs)

    def _write_test_script(self, code, filename='mytestscript.py'):
        """
        Dedents the given code (a multiline string) and writes it out to
        a file in a temporary folder like /tmp/tmpUDCn7x/mytestscript.py.
        """
        if isinstance(code, bytes):
            code = code.decode('utf-8')
        # Be explicit about encoding the temp file as UTF-8 (issue #63):
        with io.open(self.tempdir + filename, 'wt', encoding='utf-8') as f:
            f.write(dedent(code))

    def _read_test_script(self, filename='mytestscript.py'):
        with io.open(self.tempdir + filename, 'rt', encoding='utf-8') as f:
            newsource = f.read()
        return newsource

    def _futurize_test_script(self, filename='mytestscript.py', stages=(1, 2),
                              all_imports=False, from3=False,
                              conservative=False):
        params = []
        stages = list(stages)
        if all_imports:
            params.append('--all-imports')
        if from3:
            script = 'pasteurize.py'
        else:
            script = 'futurize.py'
            if stages == [1]:
                params.append('--stage1')
            elif stages == [2]:
                params.append('--stage2')
            else:
                assert stages == [1, 2]
            if conservative:
                params.append('--conservative')
            # No extra params needed

        # Absolute file path:
        fn = self.tempdir + filename
        call_args = [sys.executable, script] + params + ['-w', fn]
        try:
            output = check_output(call_args, stderr=STDOUT, env=self.env)
        except CalledProcessError as e:
            with open(fn) as f:
                msg = (
                    'Error running the command %s\n'
                    '%s\n'
                    'Contents of file %s:\n'
                    '\n'
                    '%s') % (
                        ' '.join(call_args),
                        'env=%s' % self.env,
                        fn,
                        '----\n%s\n----' % f.read(),
                    )
            ErrorClass = (FuturizeError if 'futurize' in script else PasteurizeError)

            if not hasattr(e, 'output'):
                # The attribute CalledProcessError.output doesn't exist on Py2.6
                e.output = None
            raise ErrorClass(msg, e.returncode, e.cmd, output=e.output)
        return output

    def _run_test_script(self, filename='mytestscript.py',
                         interpreter=sys.executable):
        # Absolute file path:
        fn = self.tempdir + filename
        try:
            output = check_output([interpreter, fn],
                                  env=self.env, stderr=STDOUT)
        except CalledProcessError as e:
            with open(fn) as f:
                msg = (
                    'Error running the command %s\n'
                    '%s\n'
                    'Contents of file %s:\n'
                    '\n'
                    '%s') % (
                        ' '.join([interpreter, fn]),
                        'env=%s' % self.env,
                        fn,
                        '----\n%s\n----' % f.read(),
                    )
            if not hasattr(e, 'output'):
                # The attribute CalledProcessError.output doesn't exist on Py2.6
                e.output = None
            raise VerboseCalledProcessError(msg, e.returncode, e.cmd, output=e.output)
        return output


# Decorator to skip some tests on Python 2.6 ...
skip26 = unittest.skipIf(PY26, "this test is known to fail on Py2.6")


def expectedFailurePY3(func):
    if not PY3:
        return func
    return unittest.expectedFailure(func)

def expectedFailurePY26(func):
    if not PY26:
        return func
    return unittest.expectedFailure(func)


def expectedFailurePY27(func):
    if not PY27:
        return func
    return unittest.expectedFailure(func)


def expectedFailurePY2(func):
    if not PY2:
        return func
    return unittest.expectedFailure(func)


# Renamed in Py3.3:
if not hasattr(unittest.TestCase, 'assertRaisesRegex'):
    unittest.TestCase.assertRaisesRegex = unittest.TestCase.assertRaisesRegexp

# From Py3.3:
def assertRegex(self, text, expected_regex, msg=None):
    """Fail the test unless the text matches the regular expression."""
    if isinstance(expected_regex, (str, unicode)):
        assert expected_regex, "expected_regex must not be empty."
        expected_regex = re.compile(expected_regex)
    if not expected_regex.search(text):
        msg = msg or "Regex didn't match"
        msg = '%s: %r not found in %r' % (msg, expected_regex.pattern, text)
        raise self.failureException(msg)

if not hasattr(unittest.TestCase, 'assertRegex'):
    bind_method(unittest.TestCase, 'assertRegex', assertRegex)

class _AssertRaisesBaseContext(object):

    def __init__(self, expected, test_case, callable_obj=None,
                 expected_regex=None):
        self.expected = expected
        self.test_case = test_case
        if callable_obj is not None:
            try:
                self.obj_name = callable_obj.__name__
            except AttributeError:
                self.obj_name = str(callable_obj)
        else:
            self.obj_name = None
        if isinstance(expected_regex, (bytes, str)):
            expected_regex = re.compile(expected_regex)
        self.expected_regex = expected_regex
        self.msg = None

    def _raiseFailure(self, standardMsg):
        msg = self.test_case._formatMessage(self.msg, standardMsg)
        raise self.test_case.failureException(msg)

    def handle(self, name, callable_obj, args, kwargs):
        """
        If callable_obj is None, assertRaises/Warns is being used as a
        context manager, so check for a 'msg' kwarg and return self.
        If callable_obj is not None, call it passing args and kwargs.
        """
        if callable_obj is None:
            self.msg = kwargs.pop('msg', None)
            return self
        with self:
            callable_obj(*args, **kwargs)

class _AssertWarnsContext(_AssertRaisesBaseContext):
    """A context manager used to implement TestCase.assertWarns* methods."""

    def __enter__(self):
        # The __warningregistry__'s need to be in a pristine state for tests
        # to work properly.
        for v in sys.modules.values():
            if getattr(v, '__warningregistry__', None):
                v.__warningregistry__ = {}
        self.warnings_manager = warnings.catch_warnings(record=True)
        self.warnings = self.warnings_manager.__enter__()
        warnings.simplefilter("always", self.expected)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.warnings_manager.__exit__(exc_type, exc_value, tb)
        if exc_type is not None:
            # let unexpected exceptions pass through
            return
        try:
            exc_name = self.expected.__name__
        except AttributeError:
            exc_name = str(self.expected)
        first_matching = None
        for m in self.warnings:
            w = m.message
            if not isinstance(w, self.expected):
                continue
            if first_matching is None:
                first_matching = w
            if (self.expected_regex is not None and
                not self.expected_regex.search(str(w))):
                continue
            # store warning for later retrieval
            self.warning = w
            self.filename = m.filename
            self.lineno = m.lineno
            return
        # Now we simply try to choose a helpful failure message
        if first_matching is not None:
            self._raiseFailure('"{}" does not match "{}"'.format(
                     self.expected_regex.pattern, str(first_matching)))
        if self.obj_name:
            self._raiseFailure("{} not triggered by {}".format(exc_name,
                                                               self.obj_name))
        else:
            self._raiseFailure("{} not triggered".format(exc_name))


def assertWarns(self, expected_warning, callable_obj=None, *args, **kwargs):
    """Fail unless a warning of class warnClass is triggered
       by callable_obj when invoked with arguments args and keyword
       arguments kwargs.  If a different type of warning is
       triggered, it will not be handled: depending on the other
       warning filtering rules in effect, it might be silenced, printed
       out, or raised as an exception.

       If called with callable_obj omitted or None, will return a
       context object used like this::

            with self.assertWarns(SomeWarning):
                do_something()

       An optional keyword argument 'msg' can be provided when assertWarns
       is used as a context object.

       The context manager keeps a reference to the first matching
       warning as the 'warning' attribute; similarly, the 'filename'
       and 'lineno' attributes give you information about the line
       of Python code from which the warning was triggered.
       This allows you to inspect the warning after the assertion::

           with self.assertWarns(SomeWarning) as cm:
               do_something()
           the_warning = cm.warning
           self.assertEqual(the_warning.some_attribute, 147)
    """
    context = _AssertWarnsContext(expected_warning, self, callable_obj)
    return context.handle('assertWarns', callable_obj, args, kwargs)

if not hasattr(unittest.TestCase, 'assertWarns'):
    bind_method(unittest.TestCase, 'assertWarns', assertWarns)
