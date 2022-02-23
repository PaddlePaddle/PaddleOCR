# -*- coding: utf-8 -*-
#
#  SelfTest/__init__.py: Self-test for PyCrypto
#
# Written in 2008 by Dwayne C. Litzenberger <dlitz@dlitz.net>
#
# ===================================================================
# The contents of this file are dedicated to the public domain.  To
# the extent that dedication to the public domain is not available,
# everyone is granted a worldwide, perpetual, royalty-free,
# non-exclusive license to exercise all rights associated with the
# contents of this file for any purpose whatsoever.
# No rights are reserved.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================

"""Self tests

These tests should perform quickly and can ideally be used every time an
application runs.
"""

__revision__ = "$Id$"

import sys
import unittest
from Crypto.Util.py3compat import StringIO

class SelfTestError(Exception):
    def __init__(self, message, result):
        Exception.__init__(self, message, result)
        self.message = message
        self.result = result

def run(module=None, verbosity=0, stream=None, tests=None, config=None, **kwargs):
    """Execute self-tests.

    This raises SelfTestError if any test is unsuccessful.

    You may optionally pass in a sub-module of SelfTest if you only want to
    perform some of the tests.  For example, the following would test only the
    hash modules:

        Crypto.SelfTest.run(Crypto.SelfTest.Hash)

    """

    if config is None:
        config = {}
    suite = unittest.TestSuite()
    if module is None:
        if tests is None:
            tests = get_tests(config=config)
        suite.addTests(tests)
    else:
        if tests is None:
            suite.addTests(module.get_tests(config=config))
        else:
            raise ValueError("'module' and 'tests' arguments are mutually exclusive")
    if stream is None:
        kwargs['stream'] = StringIO()
    else:
        kwargs['stream'] = stream
    runner = unittest.TextTestRunner(verbosity=verbosity, **kwargs)
    result = runner.run(suite)
    if not result.wasSuccessful():
        if stream is None:
            sys.stderr.write(kwargs['stream'].getvalue())
        raise SelfTestError("Self-test failed", result)
    return result

def get_tests(config={}):
    tests = []
    from Crypto.SelfTest import Cipher; tests += Cipher.get_tests(config=config)
    from Crypto.SelfTest import Hash;   tests += Hash.get_tests(config=config)
    from Crypto.SelfTest import Protocol; tests += Protocol.get_tests(config=config)
    from Crypto.SelfTest import PublicKey; tests += PublicKey.get_tests(config=config)
    from Crypto.SelfTest import Random; tests += Random.get_tests(config=config)
    from Crypto.SelfTest import Util;   tests += Util.get_tests(config=config)
    from Crypto.SelfTest import Signature;   tests += Signature.get_tests(config=config)
    from Crypto.SelfTest import IO;   tests += IO.get_tests(config=config)
    from Crypto.SelfTest import Math;   tests += Math.get_tests(config=config)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
