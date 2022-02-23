#
#  SelfTest/IO/test_PBES.py: Self-test for the _PBES module
#
# ===================================================================
#
# Copyright (c) 2014, Legrandin <helderijs@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# ===================================================================

"""Self-tests for Crypto.IO._PBES module"""

import unittest
from Crypto.Util.py3compat import *

from Crypto.IO._PBES import PBES2


class TestPBES2(unittest.TestCase):

    def setUp(self):
        self.ref = b("Test data")
        self.passphrase = b("Passphrase")

    def test1(self):
        ct = PBES2.encrypt(self.ref, self.passphrase,
                           'PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test2(self):
        ct = PBES2.encrypt(self.ref, self.passphrase,
                           'PBKDF2WithHMAC-SHA1AndAES128-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test3(self):
        ct = PBES2.encrypt(self.ref, self.passphrase,
                           'PBKDF2WithHMAC-SHA1AndAES192-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test4(self):
        ct = PBES2.encrypt(self.ref, self.passphrase,
                           'scryptAndAES128-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test5(self):
        ct = PBES2.encrypt(self.ref, self.passphrase,
                           'scryptAndAES192-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)

    def test6(self):
        ct = PBES2.encrypt(self.ref, self.passphrase,
                           'scryptAndAES256-CBC')
        pt = PBES2.decrypt(ct, self.passphrase)
        self.assertEqual(self.ref, pt)


def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases
    listTests = []
    listTests += list_test_cases(TestPBES2)
    return listTests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
