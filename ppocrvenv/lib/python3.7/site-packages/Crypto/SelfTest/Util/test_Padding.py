#
#  SelfTest/Util/test_Padding.py: Self-test for padding functions
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

import unittest
from binascii import unhexlify as uh

from Crypto.Util.py3compat import *
from Crypto.SelfTest.st_common import list_test_cases
from Crypto.Util.Padding import pad, unpad

class PKCS7_Tests(unittest.TestCase):

    def test1(self):
        padded = pad(b(""), 4)
        self.assertTrue(padded == uh(b("04040404")))
        padded = pad(b(""), 4, 'pkcs7')
        self.assertTrue(padded == uh(b("04040404")))
        back = unpad(padded, 4)
        self.assertTrue(back == b(""))

    def test2(self):
        padded = pad(uh(b("12345678")), 4)
        self.assertTrue(padded == uh(b("1234567804040404")))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b("12345678")))

    def test3(self):
        padded = pad(uh(b("123456")), 4)
        self.assertTrue(padded == uh(b("12345601")))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b("123456")))

    def test4(self):
        padded = pad(uh(b("1234567890")), 4)
        self.assertTrue(padded == uh(b("1234567890030303")))
        back = unpad(padded, 4)
        self.assertTrue(back == uh(b("1234567890")))

    def testn1(self):
        self.assertRaises(ValueError, pad, uh(b("12")), 4, 'pkcs8')

    def testn2(self):
        self.assertRaises(ValueError, unpad, b("\0\0\0"), 4)
        self.assertRaises(ValueError, unpad, b(""), 4)

    def testn3(self):
        self.assertRaises(ValueError, unpad, b("123456\x02"), 4)
        self.assertRaises(ValueError, unpad, b("123456\x00"), 4)
        self.assertRaises(ValueError, unpad, b("123456\x05\x05\x05\x05\x05"), 4)

class X923_Tests(unittest.TestCase):

    def test1(self):
        padded = pad(b(""), 4, 'x923')
        self.assertTrue(padded == uh(b("00000004")))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == b(""))

    def test2(self):
        padded = pad(uh(b("12345678")), 4, 'x923')
        self.assertTrue(padded == uh(b("1234567800000004")))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == uh(b("12345678")))

    def test3(self):
        padded = pad(uh(b("123456")), 4, 'x923')
        self.assertTrue(padded == uh(b("12345601")))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == uh(b("123456")))

    def test4(self):
        padded = pad(uh(b("1234567890")), 4, 'x923')
        self.assertTrue(padded == uh(b("1234567890000003")))
        back = unpad(padded, 4, 'x923')
        self.assertTrue(back == uh(b("1234567890")))

    def testn1(self):
        self.assertRaises(ValueError, unpad, b("123456\x02"), 4, 'x923')
        self.assertRaises(ValueError, unpad, b("123456\x00"), 4, 'x923')
        self.assertRaises(ValueError, unpad, b("123456\x00\x00\x00\x00\x05"), 4, 'x923')
        self.assertRaises(ValueError, unpad, b(""), 4, 'x923')

class ISO7816_Tests(unittest.TestCase):

    def test1(self):
        padded = pad(b(""), 4, 'iso7816')
        self.assertTrue(padded == uh(b("80000000")))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == b(""))

    def test2(self):
        padded = pad(uh(b("12345678")), 4, 'iso7816')
        self.assertTrue(padded == uh(b("1234567880000000")))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == uh(b("12345678")))

    def test3(self):
        padded = pad(uh(b("123456")), 4, 'iso7816')
        self.assertTrue(padded == uh(b("12345680")))
        #import pdb; pdb.set_trace()
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == uh(b("123456")))

    def test4(self):
        padded = pad(uh(b("1234567890")), 4, 'iso7816')
        self.assertTrue(padded == uh(b("1234567890800000")))
        back = unpad(padded, 4, 'iso7816')
        self.assertTrue(back == uh(b("1234567890")))

    def testn1(self):
        self.assertRaises(ValueError, unpad, b("123456\x81"), 4, 'iso7816')
        self.assertRaises(ValueError, unpad, b(""), 4, 'iso7816')

def get_tests(config={}):
    tests = []
    tests += list_test_cases(PKCS7_Tests)
    tests += list_test_cases(X923_Tests)
    tests += list_test_cases(ISO7816_Tests)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

