# -*- coding: utf-8 -*-
#
#  SelfTest/Util/__init__.py: Self-test for utility modules
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

"""Self-test for utility modules"""

__revision__ = "$Id$"

import os

def get_tests(config={}):
    tests = []
    from Crypto.SelfTest.Util import test_number; tests += test_number.get_tests(config=config)
    from Crypto.SelfTest.Util import test_Counter; tests += test_Counter.get_tests(config=config)
    from Crypto.SelfTest.Util import test_Padding; tests += test_Padding.get_tests(config=config)
    from Crypto.SelfTest.Util import test_strxor; tests += test_strxor.get_tests(config=config)
    from Crypto.SelfTest.Util import test_asn1; tests += test_asn1.get_tests(config=config)
    from Crypto.SelfTest.Util import test_rfc1751; tests += test_rfc1751.get_tests(config=config)
    return tests

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
