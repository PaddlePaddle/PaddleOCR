# -*- coding: utf-8 -*-
#
#  SelfTest/Util/test_number.py: Self-test for parts of the Crypto.Util.number module
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

"""Self-tests for (some of) Crypto.Util.number"""

import math
import unittest

from Crypto.Util.py3compat import *
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Util import number
from Crypto.Util.number import long_to_bytes


class MyError(Exception):
    """Dummy exception used for tests"""

# NB: In some places, we compare tuples instead of just output values so that
# if any inputs cause a test failure, we'll be able to tell which ones.

class MiscTests(unittest.TestCase):

    def test_ceil_div(self):
        """Util.number.ceil_div"""
        self.assertRaises(TypeError, number.ceil_div, "1", 1)
        self.assertRaises(ZeroDivisionError, number.ceil_div, 1, 0)
        self.assertRaises(ZeroDivisionError, number.ceil_div, -1, 0)

        # b = 1
        self.assertEqual(0, number.ceil_div(0, 1))
        self.assertEqual(1, number.ceil_div(1, 1))
        self.assertEqual(2, number.ceil_div(2, 1))
        self.assertEqual(3, number.ceil_div(3, 1))

        # b = 2
        self.assertEqual(0, number.ceil_div(0, 2))
        self.assertEqual(1, number.ceil_div(1, 2))
        self.assertEqual(1, number.ceil_div(2, 2))
        self.assertEqual(2, number.ceil_div(3, 2))
        self.assertEqual(2, number.ceil_div(4, 2))
        self.assertEqual(3, number.ceil_div(5, 2))

        # b = 3
        self.assertEqual(0, number.ceil_div(0, 3))
        self.assertEqual(1, number.ceil_div(1, 3))
        self.assertEqual(1, number.ceil_div(2, 3))
        self.assertEqual(1, number.ceil_div(3, 3))
        self.assertEqual(2, number.ceil_div(4, 3))
        self.assertEqual(2, number.ceil_div(5, 3))
        self.assertEqual(2, number.ceil_div(6, 3))
        self.assertEqual(3, number.ceil_div(7, 3))

        # b = 4
        self.assertEqual(0, number.ceil_div(0, 4))
        self.assertEqual(1, number.ceil_div(1, 4))
        self.assertEqual(1, number.ceil_div(2, 4))
        self.assertEqual(1, number.ceil_div(3, 4))
        self.assertEqual(1, number.ceil_div(4, 4))
        self.assertEqual(2, number.ceil_div(5, 4))
        self.assertEqual(2, number.ceil_div(6, 4))
        self.assertEqual(2, number.ceil_div(7, 4))
        self.assertEqual(2, number.ceil_div(8, 4))
        self.assertEqual(3, number.ceil_div(9, 4))

    def test_getPrime(self):
        """Util.number.getPrime"""
        self.assertRaises(ValueError, number.getPrime, -100)
        self.assertRaises(ValueError, number.getPrime, 0)
        self.assertRaises(ValueError, number.getPrime, 1)

        bits = 4
        for i in range(100):
            x = number.getPrime(bits)
            self.assertEqual(x >= (1 << bits - 1), 1)
            self.assertEqual(x < (1 << bits), 1)

        bits = 512
        x = number.getPrime(bits)
        self.assertNotEqual(x % 2, 0)
        self.assertEqual(x >= (1 << bits - 1), 1)
        self.assertEqual(x < (1 << bits), 1)

    def test_getStrongPrime(self):
        """Util.number.getStrongPrime"""
        self.assertRaises(ValueError, number.getStrongPrime, 256)
        self.assertRaises(ValueError, number.getStrongPrime, 513)
        bits = 512
        x = number.getStrongPrime(bits)
        self.assertNotEqual(x % 2, 0)
        self.assertEqual(x > (1 << bits-1)-1, 1)
        self.assertEqual(x < (1 << bits), 1)
        e = 2**16+1
        x = number.getStrongPrime(bits, e)
        self.assertEqual(number.GCD(x-1, e), 1)
        self.assertNotEqual(x % 2, 0)
        self.assertEqual(x > (1 << bits-1)-1, 1)
        self.assertEqual(x < (1 << bits), 1)
        e = 2**16+2
        x = number.getStrongPrime(bits, e)
        self.assertEqual(number.GCD((x-1)>>1, e), 1)
        self.assertNotEqual(x % 2, 0)
        self.assertEqual(x > (1 << bits-1)-1, 1)
        self.assertEqual(x < (1 << bits), 1)

    def test_isPrime(self):
        """Util.number.isPrime"""
        self.assertEqual(number.isPrime(-3), False)     # Regression test: negative numbers should not be prime
        self.assertEqual(number.isPrime(-2), False)     # Regression test: negative numbers should not be prime
        self.assertEqual(number.isPrime(1), False)      # Regression test: isPrime(1) caused some versions of PyCrypto to crash.
        self.assertEqual(number.isPrime(2), True)
        self.assertEqual(number.isPrime(3), True)
        self.assertEqual(number.isPrime(4), False)
        self.assertEqual(number.isPrime(2**1279-1), True)
        self.assertEqual(number.isPrime(-(2**1279-1)), False)     # Regression test: negative numbers should not be prime
        # test some known gmp pseudo-primes taken from
        # http://www.trnicely.net/misc/mpzspsp.html
        for composite in (43 * 127 * 211, 61 * 151 * 211, 15259 * 30517,
                          346141 * 692281, 1007119 * 2014237, 3589477 * 7178953,
                          4859419 * 9718837, 2730439 * 5460877,
                          245127919 * 490255837, 963939391 * 1927878781,
                          4186358431 * 8372716861, 1576820467 * 3153640933):
            self.assertEqual(number.isPrime(int(composite)), False)

    def test_size(self):
        self.assertEqual(number.size(2),2)
        self.assertEqual(number.size(3),2)
        self.assertEqual(number.size(0xa2),8)
        self.assertEqual(number.size(0xa2ba40),8*3)
        self.assertEqual(number.size(0xa2ba40ee07e3b2bd2f02ce227f36a195024486e49c19cb41bbbdfbba98b22b0e577c2eeaffa20d883a76e65e394c69d4b3c05a1e8fadda27edb2a42bc000fe888b9b32c22d15add0cd76b3e7936e19955b220dd17d4ea904b1ec102b2e4de7751222aa99151024c7cb41cc5ea21d00eeb41f7c800834d2c6e06bce3bce7ea9a5), 1024)
        self.assertRaises(ValueError, number.size, -1)


class LongTests(unittest.TestCase):

    def test1(self):
        self.assertEqual(long_to_bytes(0), b'\x00')
        self.assertEqual(long_to_bytes(1), b'\x01')
        self.assertEqual(long_to_bytes(0x100), b'\x01\x00')
        self.assertEqual(long_to_bytes(0xFF00000000), b'\xFF\x00\x00\x00\x00')
        self.assertEqual(long_to_bytes(0xFF00000000), b'\xFF\x00\x00\x00\x00')
        self.assertEqual(long_to_bytes(0x1122334455667788), b'\x11\x22\x33\x44\x55\x66\x77\x88')
        self.assertEqual(long_to_bytes(0x112233445566778899), b'\x11\x22\x33\x44\x55\x66\x77\x88\x99')

    def test2(self):
        self.assertEqual(long_to_bytes(0, 1), b'\x00')
        self.assertEqual(long_to_bytes(0, 2), b'\x00\x00')
        self.assertEqual(long_to_bytes(1, 3), b'\x00\x00\x01')
        self.assertEqual(long_to_bytes(65535, 2), b'\xFF\xFF')
        self.assertEqual(long_to_bytes(65536, 2), b'\x00\x01\x00\x00')
        self.assertEqual(long_to_bytes(0x100, 1), b'\x01\x00')
        self.assertEqual(long_to_bytes(0xFF00000001, 6), b'\x00\xFF\x00\x00\x00\x01')
        self.assertEqual(long_to_bytes(0xFF00000001, 8), b'\x00\x00\x00\xFF\x00\x00\x00\x01')
        self.assertEqual(long_to_bytes(0xFF00000001, 10), b'\x00\x00\x00\x00\x00\xFF\x00\x00\x00\x01')
        self.assertEqual(long_to_bytes(0xFF00000001, 11), b'\x00\x00\x00\x00\x00\x00\xFF\x00\x00\x00\x01')

    def test_err1(self):
        self.assertRaises(ValueError, long_to_bytes, -1)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(MiscTests)
    tests += list_test_cases(LongTests)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
