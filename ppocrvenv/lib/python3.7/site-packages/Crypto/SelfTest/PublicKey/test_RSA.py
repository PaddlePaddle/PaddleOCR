# -*- coding: utf-8 -*-
#
#  SelfTest/PublicKey/test_RSA.py: Self-test for the RSA primitive
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

"""Self-test suite for Crypto.PublicKey.RSA"""

__revision__ = "$Id$"

import os
import pickle
from pickle import PicklingError
from Crypto.Util.py3compat import *

import unittest
from Crypto.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex

class RSATest(unittest.TestCase):
    # Test vectors from "RSA-OAEP and RSA-PSS test vectors (.zip file)"
    #   ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
    # See RSADSI's PKCS#1 page at
    #   http://www.rsa.com/rsalabs/node.asp?id=2125

    # from oaep-int.txt

    # TODO: PyCrypto treats the message as starting *after* the leading "00"
    # TODO: That behaviour should probably be changed in the future.
    plaintext = """
           eb 7a 19 ac e9 e3 00 63 50 e3 29 50 4b 45 e2
        ca 82 31 0b 26 dc d8 7d 5c 68 f1 ee a8 f5 52 67
        c3 1b 2e 8b b4 25 1f 84 d7 e0 b2 c0 46 26 f5 af
        f9 3e dc fb 25 c9 c2 b3 ff 8a e1 0e 83 9a 2d db
        4c dc fe 4f f4 77 28 b4 a1 b7 c1 36 2b aa d2 9a
        b4 8d 28 69 d5 02 41 21 43 58 11 59 1b e3 92 f9
        82 fb 3e 87 d0 95 ae b4 04 48 db 97 2f 3a c1 4f
        7b c2 75 19 52 81 ce 32 d2 f1 b7 6d 4d 35 3e 2d
    """

    ciphertext = """
        12 53 e0 4d c0 a5 39 7b b4 4a 7a b8 7e 9b f2 a0
        39 a3 3d 1e 99 6f c8 2a 94 cc d3 00 74 c9 5d f7
        63 72 20 17 06 9e 52 68 da 5d 1c 0b 4f 87 2c f6
        53 c1 1d f8 23 14 a6 79 68 df ea e2 8d ef 04 bb
        6d 84 b1 c3 1d 65 4a 19 70 e5 78 3b d6 eb 96 a0
        24 c2 ca 2f 4a 90 fe 9f 2e f5 c9 c1 40 e5 bb 48
        da 95 36 ad 87 00 c8 4f c9 13 0a de a7 4e 55 8d
        51 a7 4d df 85 d8 b5 0d e9 68 38 d6 06 3e 09 55
    """

    modulus = """
        bb f8 2f 09 06 82 ce 9c 23 38 ac 2b 9d a8 71 f7
        36 8d 07 ee d4 10 43 a4 40 d6 b6 f0 74 54 f5 1f
        b8 df ba af 03 5c 02 ab 61 ea 48 ce eb 6f cd 48
        76 ed 52 0d 60 e1 ec 46 19 71 9d 8a 5b 8b 80 7f
        af b8 e0 a3 df c7 37 72 3e e6 b4 b7 d9 3a 25 84
        ee 6a 64 9d 06 09 53 74 88 34 b2 45 45 98 39 4e
        e0 aa b1 2d 7b 61 a5 1f 52 7a 9a 41 f6 c1 68 7f
        e2 53 72 98 ca 2a 8f 59 46 f8 e5 fd 09 1d bd cb
    """

    e = 0x11    # public exponent

    prime_factor = """
        c9 7f b1 f0 27 f4 53 f6 34 12 33 ea aa d1 d9 35
        3f 6c 42 d0 88 66 b1 d0 5a 0f 20 35 02 8b 9d 86
        98 40 b4 16 66 b4 2e 92 ea 0d a3 b4 32 04 b5 cf
        ce 33 52 52 4d 04 16 a5 a4 41 e7 00 af 46 15 03
    """

    def setUp(self):
        global RSA, Random, bytes_to_long
        from Crypto.PublicKey import RSA
        from Crypto import Random
        from Crypto.Util.number import bytes_to_long, inverse
        self.n = bytes_to_long(a2b_hex(self.modulus))
        self.p = bytes_to_long(a2b_hex(self.prime_factor))

        # Compute q, d, and u from n, e, and p
        self.q = self.n // self.p
        self.d = inverse(self.e, (self.p-1)*(self.q-1))
        self.u = inverse(self.p, self.q)    # u = e**-1 (mod q)

        self.rsa = RSA

    def test_generate_1arg(self):
        """RSA (default implementation) generated key (1 argument)"""
        rsaObj = self.rsa.generate(1024)
        self._check_private_key(rsaObj)
        self._exercise_primitive(rsaObj)
        pub = rsaObj.public_key()
        self._check_public_key(pub)
        self._exercise_public_primitive(rsaObj)

    def test_generate_2arg(self):
        """RSA (default implementation) generated key (2 arguments)"""
        rsaObj = self.rsa.generate(1024, Random.new().read)
        self._check_private_key(rsaObj)
        self._exercise_primitive(rsaObj)
        pub = rsaObj.public_key()
        self._check_public_key(pub)
        self._exercise_public_primitive(rsaObj)

    def test_generate_3args(self):
        rsaObj = self.rsa.generate(1024, Random.new().read,e=65537)
        self._check_private_key(rsaObj)
        self._exercise_primitive(rsaObj)
        pub = rsaObj.public_key()
        self._check_public_key(pub)
        self._exercise_public_primitive(rsaObj)
        self.assertEqual(65537,rsaObj.e)

    def test_construct_2tuple(self):
        """RSA (default implementation) constructed key (2-tuple)"""
        pub = self.rsa.construct((self.n, self.e))
        self._check_public_key(pub)
        self._check_encryption(pub)

    def test_construct_3tuple(self):
        """RSA (default implementation) constructed key (3-tuple)"""
        rsaObj = self.rsa.construct((self.n, self.e, self.d))
        self._check_encryption(rsaObj)
        self._check_decryption(rsaObj)

    def test_construct_4tuple(self):
        """RSA (default implementation) constructed key (4-tuple)"""
        rsaObj = self.rsa.construct((self.n, self.e, self.d, self.p))
        self._check_encryption(rsaObj)
        self._check_decryption(rsaObj)

    def test_construct_5tuple(self):
        """RSA (default implementation) constructed key (5-tuple)"""
        rsaObj = self.rsa.construct((self.n, self.e, self.d, self.p, self.q))
        self._check_private_key(rsaObj)
        self._check_encryption(rsaObj)
        self._check_decryption(rsaObj)

    def test_construct_6tuple(self):
        """RSA (default implementation) constructed key (6-tuple)"""
        rsaObj = self.rsa.construct((self.n, self.e, self.d, self.p, self.q, self.u))
        self._check_private_key(rsaObj)
        self._check_encryption(rsaObj)
        self._check_decryption(rsaObj)

    def test_construct_bad_key2(self):
        tup = (self.n, 1)
        self.assertRaises(ValueError, self.rsa.construct, tup)

        # An even modulus is wrong
        tup = (self.n+1, self.e)
        self.assertRaises(ValueError, self.rsa.construct, tup)

    def test_construct_bad_key3(self):
        tup = (self.n, self.e, self.d+1)
        self.assertRaises(ValueError, self.rsa.construct, tup)

    def test_construct_bad_key5(self):
        tup = (self.n, self.e, self.d, self.p, self.p)
        self.assertRaises(ValueError, self.rsa.construct, tup)

        tup = (self.p*self.p, self.e, self.p, self.p)
        self.assertRaises(ValueError, self.rsa.construct, tup)

        tup = (self.p*self.p, 3, self.p, self.q)
        self.assertRaises(ValueError, self.rsa.construct, tup)

    def test_construct_bad_key6(self):
        tup = (self.n, self.e, self.d, self.p, self.q, 10)
        self.assertRaises(ValueError, self.rsa.construct, tup)

        from Crypto.Util.number import inverse
        tup = (self.n, self.e, self.d, self.p, self.q, inverse(self.q, self.p))
        self.assertRaises(ValueError, self.rsa.construct, tup)

    def test_factoring(self):
        rsaObj = self.rsa.construct([self.n, self.e, self.d])
        self.assertTrue(rsaObj.p==self.p or rsaObj.p==self.q)
        self.assertTrue(rsaObj.q==self.p or rsaObj.q==self.q)
        self.assertTrue(rsaObj.q*rsaObj.p == self.n)

        self.assertRaises(ValueError, self.rsa.construct, [self.n, self.e, self.n-1])

    def test_repr(self):
        rsaObj = self.rsa.construct((self.n, self.e, self.d, self.p, self.q))
        repr(rsaObj)

    def test_serialization(self):
        """RSA keys are unpickable"""

        rsa_key = self.rsa.generate(1024)
        self.assertRaises(PicklingError, pickle.dumps, rsa_key)

    def test_raw_rsa_boundary(self):
        # The argument of every RSA raw operation (encrypt/decrypt) must be
        # non-negative and no larger than the modulus
        rsa_obj = self.rsa.generate(1024)

        self.assertRaises(ValueError, rsa_obj._decrypt, rsa_obj.n)
        self.assertRaises(ValueError, rsa_obj._encrypt, rsa_obj.n)

        self.assertRaises(ValueError, rsa_obj._decrypt, -1)
        self.assertRaises(ValueError, rsa_obj._encrypt, -1)

    def test_size(self):
        pub = self.rsa.construct((self.n, self.e))
        self.assertEqual(pub.size_in_bits(), 1024)
        self.assertEqual(pub.size_in_bytes(), 128)

    def _check_private_key(self, rsaObj):
        from Crypto.Math.Numbers import Integer

        # Check capabilities
        self.assertEqual(1, rsaObj.has_private())

        # Sanity check key data
        self.assertEqual(rsaObj.n, rsaObj.p * rsaObj.q)     # n = pq
        lcm = int(Integer(rsaObj.p-1).lcm(rsaObj.q-1))
        self.assertEqual(1, rsaObj.d * rsaObj.e % lcm) # ed = 1 (mod LCM(p-1, q-1))
        self.assertEqual(1, rsaObj.p * rsaObj.u % rsaObj.q) # pu = 1 (mod q)
        self.assertEqual(1, rsaObj.p > 1)   # p > 1
        self.assertEqual(1, rsaObj.q > 1)   # q > 1
        self.assertEqual(1, rsaObj.e > 1)   # e > 1
        self.assertEqual(1, rsaObj.d > 1)   # d > 1

    def _check_public_key(self, rsaObj):
        ciphertext = a2b_hex(self.ciphertext)

        # Check capabilities
        self.assertEqual(0, rsaObj.has_private())

        # Check rsaObj.[ne] -> rsaObj.[ne] mapping
        self.assertEqual(rsaObj.n, rsaObj.n)
        self.assertEqual(rsaObj.e, rsaObj.e)

        # Check that private parameters are all missing
        self.assertEqual(0, hasattr(rsaObj, 'd'))
        self.assertEqual(0, hasattr(rsaObj, 'p'))
        self.assertEqual(0, hasattr(rsaObj, 'q'))
        self.assertEqual(0, hasattr(rsaObj, 'u'))

        # Sanity check key data
        self.assertEqual(1, rsaObj.e > 1)   # e > 1

        # Public keys should not be able to sign or decrypt
        self.assertRaises(TypeError, rsaObj._decrypt,
                bytes_to_long(ciphertext))

        # Check __eq__ and __ne__
        self.assertEqual(rsaObj.public_key() == rsaObj.public_key(),True) # assert_
        self.assertEqual(rsaObj.public_key() != rsaObj.public_key(),False) # assertFalse

        self.assertEqual(rsaObj.publickey(), rsaObj.public_key())

    def _exercise_primitive(self, rsaObj):
        # Since we're using a randomly-generated key, we can't check the test
        # vector, but we can make sure encryption and decryption are inverse
        # operations.
        ciphertext = bytes_to_long(a2b_hex(self.ciphertext))

        # Test decryption
        plaintext = rsaObj._decrypt(ciphertext)

        # Test encryption (2 arguments)
        new_ciphertext2 = rsaObj._encrypt(plaintext)
        self.assertEqual(ciphertext, new_ciphertext2)

    def _exercise_public_primitive(self, rsaObj):
        plaintext = a2b_hex(self.plaintext)

        # Test encryption (2 arguments)
        new_ciphertext2 = rsaObj._encrypt(bytes_to_long(plaintext))

    def _check_encryption(self, rsaObj):
        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)

        # Test encryption
        new_ciphertext2 = rsaObj._encrypt(bytes_to_long(plaintext))
        self.assertEqual(bytes_to_long(ciphertext), new_ciphertext2)

    def _check_decryption(self, rsaObj):
        plaintext = bytes_to_long(a2b_hex(self.plaintext))
        ciphertext = bytes_to_long(a2b_hex(self.ciphertext))

        # Test plain decryption
        new_plaintext = rsaObj._decrypt(ciphertext)
        self.assertEqual(plaintext, new_plaintext)


def get_tests(config={}):
    tests = []
    tests += list_test_cases(RSATest)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
