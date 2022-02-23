#
#  SelfTest/Protocol/test_secret_sharing.py: Self-test for secret sharing protocols
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

from unittest import main, TestCase, TestSuite
from binascii import unhexlify, hexlify

from Crypto.Util.py3compat import *
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Protocol.SecretSharing import Shamir, _Element, \
                                          _mult_gf2, _div_gf2

class GF2_Tests(TestCase):

    def test_mult_gf2(self):
        # Prove mult by zero
        x = _mult_gf2(0,0)
        self.assertEqual(x, 0)

        # Prove mult by unity
        x = _mult_gf2(34, 1)
        self.assertEqual(x, 34)

        z = 3                       # (x+1)
        y = _mult_gf2(z, z)
        self.assertEqual(y, 5)      # (x+1)^2 = x^2 + 1
        y = _mult_gf2(y, z)
        self.assertEqual(y, 15)     # (x+1)^3 = x^3 + x^2 + x + 1
        y = _mult_gf2(y, z)
        self.assertEqual(y, 17)     # (x+1)^4 = x^4 + 1

        # Prove linearity works
        comps = [1, 4, 128, 2**34]
        sum_comps = 1+4+128+2**34
        y = 908
        z = _mult_gf2(sum_comps, y)
        w = 0
        for x in comps:
            w ^= _mult_gf2(x, y)
        self.assertEqual(w, z)

    def test_div_gf2(self):
        from Crypto.Util.number import size as deg

        x, y = _div_gf2(567, 7)
        self.assertTrue(deg(y) < deg(7))

        w = _mult_gf2(x, 7) ^ y
        self.assertEqual(567, w)

        x, y = _div_gf2(7, 567)
        self.assertEqual(x, 0)
        self.assertEqual(y, 7)

class Element_Tests(TestCase):

    def test1(self):
        # Test encondings
        e = _Element(256)
        self.assertEqual(int(e), 256)
        self.assertEqual(e.encode(), bchr(0)*14 + b("\x01\x00"))

        e = _Element(bchr(0)*14 + b("\x01\x10"))
        self.assertEqual(int(e), 0x110)
        self.assertEqual(e.encode(), bchr(0)*14 + b("\x01\x10"))

        # Only 16 byte string are a valid encoding
        self.assertRaises(ValueError, _Element, bchr(0))

    def test2(self):
        # Test addition
        e = _Element(0x10)
        f = _Element(0x0A)
        self.assertEqual(int(e+f), 0x1A)

    def test3(self):
        # Test multiplication
        zero = _Element(0)
        one = _Element(1)
        two = _Element(2)

        x = _Element(6) * zero
        self.assertEqual(int(x), 0)

        x = _Element(6) * one
        self.assertEqual(int(x), 6)

        x = _Element(2**127) * two
        self.assertEqual(int(x), 1 + 2 + 4 + 128)

    def test4(self):
        # Test inversion
        one = _Element(1)

        x = one.inverse()
        self.assertEqual(int(x), 1)

        x = _Element(82323923)
        y = x.inverse()
        self.assertEqual(int(x * y), 1)

class Shamir_Tests(TestCase):

    def test1(self):
        # Test splitting
        shares = Shamir.split(2, 3, bchr(90)*16)
        self.assertEqual(len(shares), 3)
        for index in range(3):
            self.assertEqual(shares[index][0], index+1)
            self.assertEqual(len(shares[index][1]), 16)

    def test2(self):
        # Test recombine
        from itertools import permutations

        test_vectors = (
            (2, "d9fe73909bae28b3757854c0af7ad405",
             "1-594ae8964294174d95c33756d2504170",
             "2-d897459d29da574eb40e93ec552ffe6e",
             "3-5823de9bf0e068b054b5f07a28056b1b",
             "4-db2c1f8bff46d748f795da995bd080cb"),
            (2, "bf4f902d9a7efafd1f3ffd9291fd5de9",
             "1-557bd3b0748064b533469722d1cc7935",
             "2-6b2717164783c66d47cd28f2119f14d0",
             "3-8113548ba97d58256bb4424251ae300c",
             "4-179e9e5a218483ddaeda57539139cf04"),
            (3, "ec96aa5c14c9faa699354cf1da74e904",
             "1-64579fbf1908d66f7239bf6e2b4e41e1",
             "2-6cd9428df8017b52322561e8c672ae3e",
             "3-e418776ef5c0579bd9299277374806dd",
             "4-ab3f77a0107398d23b323e581bb43f5d",
             "5-23fe42431db2b41bd03ecdc7ea8e97ac"),
            (3, "44cf249b68b80fcdc27b47be60c2c145",
             "1-d6515a3905cd755119b86e311c801e31",
             "2-16693d9ac9f10c254036ced5f8917fa3",
             "3-84f74338a48476b99bf5e75a84d3a0d1",
             "4-3fe8878dc4a5d35811cf3cbcd33dbe52",
             "5-ad76f92fa9d0a9c4ca0c1533af7f6132"),
            (5, "5398717c982db935d968eebe53a47f5a",
             "1-be7be2dd4c068e7ef576aaa1b1c11b01",
             "2-f821f5848441cb98b3eb467e2733ee21",
             "3-25ee52f53e203f6e29a0297b5ab486b5",
             "4-fc9fb58ef74dab947fbf9acd9d5d83cd",
             "5-b1949cce46d81552e65f248d3f74cc5c",
             "6-d64797f59977c4d4a7956ad916da7699",
             "7-ab608a6546a8b9af8820ff832b1135c7"),
            (5, "4a78db90fbf35da5545d2fb728e87596",
             "1-08daf9a25d8aa184cfbf02b30a0ed6a0",
             "2-dda28261e36f0b14168c2cf153fb734e",
             "3-e9fdec5505d674a57f9836c417c1ecaa",
             "4-4dce5636ae06dee42d2c82e65f06c735",
             "5-3963dc118afc2ba798fa1d452b28ef00",
             "6-6dfe6ff5b09e94d2f84c382b12f42424",
             "7-6faea9d4d4a4e201bf6c90b9000630c3"),
            (10, "eccbf6d66d680b49b073c4f1ddf804aa",
             "01-7d8ac32fe4ae209ead1f3220fda34466",
             "02-f9144e76988aad647d2e61353a6e96d5",
             "03-b14c3b80179203363922d60760271c98",
             "04-770bb2a8c28f6cee89e00f4d5cc7f861",
             "05-6e3d7073ea368334ef67467871c66799",
             "06-248792bc74a98ce024477c13c8fb5f8d",
             "07-fcea4640d2db820c0604851e293d2487",
             "08-2776c36fb714bb1f8525a0be36fc7dba",
             "09-6ee7ac8be773e473a4bf75ee5f065762",
             "10-33657fc073354cf91d4a68c735aacfc8",
             "11-7645c65094a5868bf225c516fdee2d0c",
             "12-840485aacb8226631ecd9c70e3018086"),
            (10, "377e63bdbb5f7d4dc58a483d035212bb",
             "01-32c53260103be431c843b1a633afe3bd",
             "02-0107eb16cb8695084d452d2cc50bc7d6",
             "03-df1e5c66cd755287fb0446faccd72a06",
             "04-361bbcd5d40797f49dfa1898652da197",
             "05-160d3ad1512f7dec7fd9344aed318591",
             "06-659af6d95df4f25beca4fb9bfee3b7e8",
             "07-37f3b208977bad50b3724566b72bfa9d",
             "08-6c1de2dfc69c2986142c26a8248eb316",
             "09-5e19220837a396bd4bc8cd685ff314c3",
             "10-86e7b864fb0f3d628e46d50c1ba92f1c",
             "11-065d0082c80b1aea18f4abe0c49df72e",
             "12-84a09430c1d20ea9f388f3123c3733a3"),
        )

        def get_share(p):
            pos = p.find('-')
            return int(p[:pos]), unhexlify(p[pos + 1:])

        for tv in test_vectors:
            k = tv[0]
            secret = unhexlify(tv[1])
            max_perms = 10
            for perm, shares_idx in enumerate(permutations(range(2, len(tv)), k)):
                if perm > max_perms:
                    break
                shares = [ get_share(tv[x]) for x in shares_idx ]
                result = Shamir.combine(shares, True)
                self.assertEqual(secret, result)

    def test3(self):
        # Loopback split/recombine
        secret = unhexlify(b("000102030405060708090a0b0c0d0e0f"))

        shares = Shamir.split(2, 3, secret)

        secret2 = Shamir.combine(shares[:2])
        self.assertEqual(secret, secret2)

        secret3 = Shamir.combine([ shares[0], shares[2] ])
        self.assertEqual(secret, secret3)

    def test4(self):
        # Loopback split/recombine (SSSS)
        secret = unhexlify(b("000102030405060708090a0b0c0d0e0f"))

        shares = Shamir.split(2, 3, secret, ssss=True)

        secret2 = Shamir.combine(shares[:2], ssss=True)
        self.assertEqual(secret, secret2)

    def test5(self):
        # Detect duplicate shares
        secret = unhexlify(b("000102030405060708090a0b0c0d0e0f"))

        shares = Shamir.split(2, 3, secret)
        self.assertRaises(ValueError, Shamir.combine, (shares[0], shares[0]))


def get_tests(config={}):
    tests = []
    tests += list_test_cases(GF2_Tests)
    tests += list_test_cases(Element_Tests)
    tests += list_test_cases(Shamir_Tests)
    return tests

if __name__ == '__main__':
    suite = lambda: TestSuite(get_tests())
    main(defaultTest='suite')

