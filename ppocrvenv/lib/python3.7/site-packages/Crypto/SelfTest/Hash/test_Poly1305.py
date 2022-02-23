#
#  SelfTest/Hash/test_Poly1305.py: Self-test for the Poly1305 module
#
# ===================================================================
#
# Copyright (c) 2018, Helder Eijs <helderijs@gmail.com>
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

"""Self-test suite for Crypto.Hash._Poly1305"""

import json
import unittest
from binascii import unhexlify, hexlify

from .common import make_mac_tests
from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import Poly1305
from Crypto.Cipher import AES, ChaCha20

from Crypto.Util.py3compat import tobytes
from Crypto.Util.strxor import strxor_c

# This is a list of (r+s keypair, data, result, description, keywords) tuples.
test_data_basic = [
    (
        "85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b",
        hexlify(b"Cryptographic Forum Research Group").decode(),
        "a8061dc1305136c6c22b8baf0c0127a9",
        "RFC7539"
    ),
    (
        "746869732069732033322d62797465206b657920666f7220506f6c7931333035",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "49ec78090e481ec6c26b33b91ccc0307",
        "https://tools.ietf.org/html/draft-agl-tls-chacha20poly1305-00#section-7 A",
    ),
    (
        "746869732069732033322d62797465206b657920666f7220506f6c7931333035",
        "48656c6c6f20776f726c6421",
        "a6f745008f81c916a20dcc74eef2b2f0",
        "https://tools.ietf.org/html/draft-agl-tls-chacha20poly1305-00#section-7 B",
    ),
    (
        "746869732069732033322d62797465206b657920666f7220506f6c7931333035",
        "",
        "6b657920666f7220506f6c7931333035",
        "Generated with pure Python",
    ),
    (
        "746869732069732033322d62797465206b657920666f7220506f6c7931333035",
        "FF",
        "f7e4e0ef4c46d106219da3d1bdaeb3ff",
        "Generated with pure Python",
    ),
    (
        "746869732069732033322d62797465206b657920666f7220506f6c7931333035",
        "FF00",
        "7471eceeb22988fc936da1d6e838b70e",
        "Generated with pure Python",
    ),
    (
        "746869732069732033322d62797465206b657920666f7220506f6c7931333035",
        "AA" * 17,
        "32590bc07cb2afaccca3f67f122975fe",
        "Generated with pure Python",
    ),
    (
        "00" * 32,
        "00" * 64,
        "00" * 16,
        "RFC7539 A.3 #1",
    ),
    (
        "0000000000000000000000000000000036e5f6b5c5e06070f0efca96227a863e",
        hexlify(
        b"Any submission t"
        b"o the IETF inten"
        b"ded by the Contr"
        b"ibutor for publi"
        b"cation as all or"
        b" part of an IETF"
        b" Internet-Draft "
        b"or RFC and any s"
        b"tatement made wi"
        b"thin the context"
        b" of an IETF acti"
        b"vity is consider"
        b"ed an \"IETF Cont"
        b"ribution\". Such "
        b"statements inclu"
        b"de oral statemen"
        b"ts in IETF sessi"
        b"ons, as well as "
        b"written and elec"
        b"tronic communica"
        b"tions made at an"
        b"y time or place,"
        b" which are addre"
        b"ssed to").decode(),
        "36e5f6b5c5e06070f0efca96227a863e",
        "RFC7539 A.3 #2",
    ),
    (
        "36e5f6b5c5e06070f0efca96227a863e00000000000000000000000000000000",
        hexlify(
        b"Any submission t"
        b"o the IETF inten"
        b"ded by the Contr"
        b"ibutor for publi"
        b"cation as all or"
        b" part of an IETF"
        b" Internet-Draft "
        b"or RFC and any s"
        b"tatement made wi"
        b"thin the context"
        b" of an IETF acti"
        b"vity is consider"
        b"ed an \"IETF Cont"
        b"ribution\". Such "
        b"statements inclu"
        b"de oral statemen"
        b"ts in IETF sessi"
        b"ons, as well as "
        b"written and elec"
        b"tronic communica"
        b"tions made at an"
        b"y time or place,"
        b" which are addre"
        b"ssed to").decode(),
        "f3477e7cd95417af89a6b8794c310cf0",
        "RFC7539 A.3 #3",
    ),
    (
        "1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0",
        "2754776173206272696c6c69672c2061"
        "6e642074686520736c6974687920746f"
        "7665730a446964206779726520616e64"
        "2067696d626c6520696e207468652077"
        "6162653a0a416c6c206d696d73792077"
        "6572652074686520626f726f676f7665"
        "732c0a416e6420746865206d6f6d6520"
        "7261746873206f757467726162652e",
        "4541669a7eaaee61e708dc7cbcc5eb62",
        "RFC7539 A.3 #4",
    ),
    (
        "02" + "00" * 31,
        "FF" * 16,
        "03" + "00" * 15,
        "RFC7539 A.3 #5",
    ),
    (
        "02" + "00" * 15 + "FF" * 16,
        "02" + "00" * 15,
        "03" + "00" * 15,
        "RFC7539 A.3 #6",
    ),
    (
        "01" + "00" * 31,
        "FF" * 16 + "F0" + "FF" * 15 + "11" + "00" * 15,
        "05" + "00" * 15,
        "RFC7539 A.3 #7",
    ),
    (
        "01" + "00" * 31,
        "FF" * 16 + "FB" + "FE" * 15 + "01" * 16,
        "00" * 16,
        "RFC7539 A.3 #8",
    ),
    (
        "02" + "00" * 31,
        "FD" + "FF" * 15,
        "FA" + "FF" * 15,
        "RFC7539 A.3 #9",
    ),
    (
        "01 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
        "E3 35 94 D7 50 5E 43 B9 00 00 00 00 00 00 00 00"
        "33 94 D7 50 5E 43 79 CD 01 00 00 00 00 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00"
        "01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
        "14 00 00 00 00 00 00 00 55 00 00 00 00 00 00 00",
        "RFC7539 A.3 #10",
    ),
    (
        "01 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
        "E3 35 94 D7 50 5E 43 B9 00 00 00 00 00 00 00 00"
        "33 94 D7 50 5E 43 79 CD 01 00 00 00 00 00 00 00"
        "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
        "13" + "00" * 15,
        "RFC7539 A.3 #11",
    ),
]

# This is a list of (key(k+r), data, result, description, keywords) tuples.
test_data_aes = [
    (
        "ec074c835580741701425b623235add6851fc40c3467ac0be05cc20404f3f700",
        "f3f6",
        "f4c633c3044fc145f84f335cb81953de",
        "http://cr.yp.to/mac/poly1305-20050329.pdf",
        { 'cipher':AES, 'nonce':unhexlify("fb447350c4e868c52ac3275cf9d4327e") }
    ),
    (
        "75deaa25c09f208e1dc4ce6b5cad3fbfa0f3080000f46400d0c7e9076c834403",
        "",
        "dd3fab2251f11ac759f0887129cc2ee7",
        "http://cr.yp.to/mac/poly1305-20050329.pdf",
        { 'cipher':AES, 'nonce':unhexlify("61ee09218d29b0aaed7e154a2c5509cc") }
    ),
    (
        "6acb5f61a7176dd320c5c1eb2edcdc7448443d0bb0d21109c89a100b5ce2c208",
        "663cea190ffb83d89593f3f476b6bc24"
        "d7e679107ea26adb8caf6652d0656136",
        "0ee1c16bb73f0f4fd19881753c01cdbe",
        "http://cr.yp.to/mac/poly1305-20050329.pdf",
        { 'cipher':AES, 'nonce':unhexlify("ae212a55399729595dea458bc621ff0e") }
    ),
    (
        "e1a5668a4d5b66a5f68cc5424ed5982d12976a08c4426d0ce8a82407c4f48207",
        "ab0812724a7f1e342742cbed374d94d1"
        "36c6b8795d45b3819830f2c04491faf0"
        "990c62e48b8018b2c3e4a0fa3134cb67"
        "fa83e158c994d961c4cb21095c1bf9",
        "5154ad0d2cb26e01274fc51148491f1b",
        "http://cr.yp.to/mac/poly1305-20050329.pdf",
        { 'cipher':AES, 'nonce':unhexlify("9ae831e743978d3a23527c7128149e3a") }
    ),
]

test_data_chacha20 = [
    (
        "00" * 32,
        "FF" * 15,
        "13cc5bbadc36b03a5163928f0bcb65aa",
        "RFC7539 A.4 #1",
        { 'cipher':ChaCha20, 'nonce':unhexlify("00" * 12) }
    ),
    (
        "00" * 31 + "01",
        "FF" * 15,
        "0baf33c1d6df211bdd50a6767e98e00a",
        "RFC7539 A.4 #2",
        { 'cipher':ChaCha20, 'nonce':unhexlify("00" * 11 + "02") }
    ),
    (
        "1c 92 40 a5 eb 55 d3 8a f3 33 88 86 04 f6 b5 f0"
        "47 39 17 c1 40 2b 80 09 9d ca 5c bc 20 70 75 c0",
        "FF" * 15,
        "e8b4c6db226cd8939e65e02eebf834ce",
        "RFC7539 A.4 #3",
        { 'cipher':ChaCha20, 'nonce':unhexlify("00" * 11 + "02") }
    ),
    (
        "1c 92 40 a5 eb 55 d3 8a f3 33 88 86 04 f6 b5 f0"
        "47 39 17 c1 40 2b 80 09 9d ca 5c bc 20 70 75 c0",
        "f3 33 88 86 00 00 00 00 00 00 4e 91 00 00 00 00"
        "64 a0 86 15 75 86 1a f4 60 f0 62 c7 9b e6 43 bd"
        "5e 80 5c fd 34 5c f3 89 f1 08 67 0a c7 6c 8c b2"
        "4c 6c fc 18 75 5d 43 ee a0 9e e9 4e 38 2d 26 b0"
        "bd b7 b7 3c 32 1b 01 00 d4 f0 3b 7f 35 58 94 cf"
        "33 2f 83 0e 71 0b 97 ce 98 c8 a8 4a bd 0b 94 81"
        "14 ad 17 6e 00 8d 33 bd 60 f9 82 b1 ff 37 c8 55"
        "97 97 a0 6e f4 f0 ef 61 c1 86 32 4e 2b 35 06 38"
        "36 06 90 7b 6a 7c 02 b0 f9 f6 15 7b 53 c8 67 e4"
        "b9 16 6c 76 7b 80 4d 46 a5 9b 52 16 cd e7 a4 e9"
        "90 40 c5 a4 04 33 22 5e e2 82 a1 b0 a0 6c 52 3e"
        "af 45 34 d7 f8 3f a1 15 5b 00 47 71 8c bc 54 6a"
        "0d 07 2b 04 b3 56 4e ea 1b 42 22 73 f5 48 27 1a"
        "0b b2 31 60 53 fa 76 99 19 55 eb d6 31 59 43 4e"
        "ce bb 4e 46 6d ae 5a 10 73 a6 72 76 27 09 7a 10"
        "49 e6 17 d9 1d 36 10 94 fa 68 f0 ff 77 98 71 30"
        "30 5b ea ba 2e da 04 df 99 7b 71 4d 6c 6f 2c 29"
        "a6 ad 5c b4 02 2b 02 70 9b 00 00 00 00 00 00 00"
        "0c 00 00 00 00 00 00 00 09 01 00 00 00 00 00 00",
        "ee ad 9d 67 89 0c bb 22 39 23 36 fe a1 85 1f 38",
        "RFC7539 A.5",
        { 'cipher':ChaCha20, 'nonce':unhexlify("000000000102030405060708") }
    ),
]


class Poly1305Test_AES(unittest.TestCase):

    key = b'\x11' * 32

    def test_new_positive(self):

        data = b'r' * 100

        h1 = Poly1305.new(key=self.key, cipher=AES)
        self.assertEqual(h1.digest_size, 16)
        self.assertEqual(len(h1.nonce), 16)
        d1 = h1.update(data).digest()
        self.assertEqual(len(d1), 16)

        h2 = Poly1305.new(key=self.key, nonce=h1.nonce, data=data, cipher=AES)
        d2 = h2.digest()
        self.assertEqual(h1.nonce, h2.nonce)
        self.assertEqual(d1, d2)

    def test_new_negative(self):
        from Crypto.Cipher import DES3

        self.assertRaises(ValueError, Poly1305.new, key=self.key[:31], cipher=AES)
        self.assertRaises(ValueError, Poly1305.new, key=self.key, cipher=DES3)
        self.assertRaises(ValueError, Poly1305.new, key=self.key, nonce=b'1' * 15, cipher=AES)
        self.assertRaises(TypeError, Poly1305.new, key=u"2" * 32, cipher=AES)
        self.assertRaises(TypeError, Poly1305.new, key=self.key, data=u"2" * 100, cipher=AES)

    def test_update(self):
        pieces = [b"\x0A" * 200, b"\x14" * 300]
        h1 = Poly1305.new(key=self.key, cipher=AES)
        h1.update(pieces[0]).update(pieces[1])
        d1 = h1.digest()

        h2 = Poly1305.new(key=self.key, cipher=AES, nonce=h1.nonce)
        h2.update(pieces[0] + pieces[1])
        d2 = h2.digest()
        self.assertEqual(d1, d2)

    def test_update_negative(self):
        h = Poly1305.new(key=self.key, cipher=AES)
        self.assertRaises(TypeError, h.update, u"string")

    def test_digest(self):
        h = Poly1305.new(key=self.key, cipher=AES)
        digest = h.digest()

        # hexdigest does not change the state
        self.assertEqual(h.digest(), digest)
        # digest returns a byte string
        self.assertTrue(isinstance(digest, type(b"digest")))

    def test_update_after_digest(self):
        msg=b"rrrrttt"

        # Normally, update() cannot be done after digest()
        h = Poly1305.new(key=self.key, data=msg[:4], cipher=AES)
        h.digest()
        self.assertRaises(TypeError, h.update, msg[4:])

    def test_hex_digest(self):
        mac = Poly1305.new(key=self.key, cipher=AES)
        digest = mac.digest()
        hexdigest = mac.hexdigest()

        # hexdigest is equivalent to digest
        self.assertEqual(hexlify(digest), tobytes(hexdigest))
        # hexdigest does not change the state
        self.assertEqual(mac.hexdigest(), hexdigest)
        # hexdigest returns a string
        self.assertTrue(isinstance(hexdigest, type("digest")))

    def test_verify(self):
        h = Poly1305.new(key=self.key, cipher=AES)
        mac = h.digest()
        h.verify(mac)
        wrong_mac = strxor_c(mac, 255)
        self.assertRaises(ValueError, h.verify, wrong_mac)

    def test_hexverify(self):
        h = Poly1305.new(key=self.key, cipher=AES)
        mac = h.hexdigest()
        h.hexverify(mac)
        self.assertRaises(ValueError, h.hexverify, "4556")

    def test_bytearray(self):

        data = b"\x00\x01\x02"
        h0 = Poly1305.new(key=self.key, data=data, cipher=AES)
        d_ref = h0.digest()

        # Data and key can be a bytearray (during initialization)
        key_ba = bytearray(self.key)
        data_ba = bytearray(data)

        h1 = Poly1305.new(key=self.key, data=data, cipher=AES, nonce=h0.nonce)
        h2 = Poly1305.new(key=key_ba, data=data_ba, cipher=AES, nonce=h0.nonce)
        key_ba[:1] = b'\xFF'
        data_ba[:1] = b'\xEE'

        self.assertEqual(h1.digest(), d_ref)
        self.assertEqual(h2.digest(), d_ref)

        # Data can be a bytearray (during operation)
        data_ba = bytearray(data)

        h1 = Poly1305.new(key=self.key, cipher=AES)
        h2 = Poly1305.new(key=self.key, cipher=AES, nonce=h1.nonce)
        h1.update(data)
        h2.update(data_ba)
        data_ba[:1] = b'\xFF'

        self.assertEqual(h1.digest(), h2.digest())

    def test_memoryview(self):

        data = b"\x00\x01\x02"

        def get_mv_ro(data):
            return memoryview(data)

        def get_mv_rw(data):
            return memoryview(bytearray(data))

        for get_mv in (get_mv_ro, get_mv_rw):

            # Data and key can be a memoryview (during initialization)
            key_mv = get_mv(self.key)
            data_mv = get_mv(data)

            h1 = Poly1305.new(key=self.key, data=data, cipher=AES)
            h2 = Poly1305.new(key=key_mv, data=data_mv, cipher=AES,
                              nonce=h1.nonce)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'
                key_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())

            # Data can be a memoryview (during operation)
            data_mv = get_mv(data)

            h1 = Poly1305.new(key=self.key, cipher=AES)
            h2 = Poly1305.new(key=self.key, cipher=AES, nonce=h1.nonce)
            h1.update(data)
            h2.update(data_mv)
            if not data_mv.readonly:
                data_mv[:1] = b'\xFF'

            self.assertEqual(h1.digest(), h2.digest())


class Poly1305Test_ChaCha20(unittest.TestCase):

    key = b'\x11' * 32

    def test_new_positive(self):
        data = b'r' * 100

        h1 = Poly1305.new(key=self.key, cipher=ChaCha20)
        self.assertEqual(h1.digest_size, 16)
        self.assertEqual(len(h1.nonce), 12)
        
        h2 = Poly1305.new(key=self.key, cipher=ChaCha20, nonce = b'8' * 8)
        self.assertEqual(len(h2.nonce), 8)
        self.assertEqual(h2.nonce, b'8' * 8)

    def test_new_negative(self):

        self.assertRaises(ValueError, Poly1305.new, key=self.key, nonce=b'1' * 7, cipher=ChaCha20)


#
# make_mac_tests() expect a new() function with signature new(key, data,
# **kwargs), and we need to adapt Poly1305's, as it only uses keywords
#
class Poly1305_New(object):

    @staticmethod
    def new(key, *data, **kwds):
        _kwds = dict(kwds)
        if len(data) == 1:
            _kwds['data'] = data[0]
        _kwds['key'] = key
        return Poly1305.new(**_kwds)


class Poly1305_Basic(object):

    @staticmethod
    def new(key, *data, **kwds):
        from Crypto.Hash.Poly1305 import Poly1305_MAC

        if len(data) == 1:
            msg = data[0]
        else:
            msg = None

        return Poly1305_MAC(key[:16], key[16:], msg)


class Poly1305AES_MC(unittest.TestCase):

    def runTest(self):
        tag = unhexlify(b"fb447350c4e868c52ac3275cf9d4327e")

        msg = b''
        for msg_len in range(5000 + 1):
            key = tag + strxor_c(tag, 0xFF)
            nonce = tag[::-1]
            if msg_len > 0:
                msg = msg + tobytes(tag[0])
            auth = Poly1305.new(key=key, nonce=nonce, cipher=AES, data=msg)
            tag = auth.digest()

        # Compare against output of original DJB's poly1305aes-20050218
        self.assertEqual("CDFA436DDD629C7DC20E1128530BAED2", auth.hexdigest().upper())


def get_tests(config={}):
    tests = make_mac_tests(Poly1305_Basic, "Poly1305", test_data_basic)
    tests += make_mac_tests(Poly1305_New, "Poly1305", test_data_aes)
    tests += make_mac_tests(Poly1305_New, "Poly1305", test_data_chacha20)
    tests += [ Poly1305AES_MC() ]
    tests += list_test_cases(Poly1305Test_AES)
    tests += list_test_cases(Poly1305Test_ChaCha20)
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
