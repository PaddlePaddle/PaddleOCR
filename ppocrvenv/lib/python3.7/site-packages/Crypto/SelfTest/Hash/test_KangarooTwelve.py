# ===================================================================
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

"""Self-test suite for Crypto.Hash.KangarooTwelve"""

import unittest
from binascii import unhexlify

from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Hash import KangarooTwelve as K12
from Crypto.Util.py3compat import b, bchr


class KangarooTwelveTest(unittest.TestCase):

    def test_length_encode(self):
        self.assertEqual(K12._length_encode(0), b'\x00')
        self.assertEqual(K12._length_encode(12), b'\x0C\x01')
        self.assertEqual(K12._length_encode(65538), b'\x01\x00\x02\x03')

    def test_new_positive(self):

        xof1 = K12.new()
        xof2 = K12.new(data=b("90"))
        xof3 = K12.new().update(b("90"))

        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

        xof1 = K12.new()
        ref = xof1.read(10)
        xof2 = K12.new(custom=b(""))
        xof3 = K12.new(custom=b("foo"))

        self.assertEqual(ref, xof2.read(10))
        self.assertNotEqual(ref, xof3.read(10))

        xof1 = K12.new(custom=b("foo"))
        xof2 = K12.new(custom=b("foo"), data=b("90"))
        xof3 = K12.new(custom=b("foo")).update(b("90"))

        self.assertNotEqual(xof1.read(10), xof2.read(10))
        xof3.read(10)
        self.assertEqual(xof2.read(10), xof3.read(10))

    def test_update(self):
        pieces = [bchr(10) * 200, bchr(20) * 300]
        h = K12.new()
        h.update(pieces[0]).update(pieces[1])
        digest = h.read(10)
        h = K12.new()
        h.update(pieces[0] + pieces[1])
        self.assertEqual(h.read(10), digest)

    def test_update_negative(self):
        h = K12.new()
        self.assertRaises(TypeError, h.update, u"string")

    def test_digest(self):
        h = K12.new()
        digest = h.read(90)

        # read returns a byte string of the right length
        self.assertTrue(isinstance(digest, type(b("digest"))))
        self.assertEqual(len(digest), 90)

    def test_update_after_read(self):
        mac = K12.new()
        mac.update(b("rrrr"))
        mac.read(90)
        self.assertRaises(TypeError, mac.update, b("ttt"))


def txt2bin(txt):
    clean = txt.replace(" ", "").replace("\n", "").replace("\r", "")
    return unhexlify(clean)


def ptn(n):
    res = bytearray(n)
    pattern = b"".join([bchr(x) for x in range(0, 0xFB)])
    for base in range(0, n - 0xFB, 0xFB):
        res[base:base + 0xFB] = pattern
    remain = n % 0xFB
    if remain:
        base = (n // 0xFB) * 0xFB
        res[base:] = pattern[:remain]
    assert(len(res) == n)
    return res


def chunked(source, size):
    for i in range(0, len(source), size):
        yield source[i:i+size]


# https://github.com/XKCP/XKCP/blob/master/tests/TestVectors/KangarooTwelve.txt
class KangarooTwelveTV(unittest.TestCase):

    def test_zero_1(self):
        tv = """1A C2 D4 50 FC 3B 42 05 D1 9D A7 BF CA 1B 37 51
             3C 08 03 57 7A C7 16 7F 06 FE 2C E1 F0 EF 39 E5"""

        btv = txt2bin(tv)
        res = K12.new().read(32)
        self.assertEqual(res, btv)

    def test_zero_2(self):
        tv = """1A C2 D4 50 FC 3B 42 05 D1 9D A7 BF CA 1B 37 51
        3C 08 03 57 7A C7 16 7F 06 FE 2C E1 F0 EF 39 E5
        42 69 C0 56 B8 C8 2E 48 27 60 38 B6 D2 92 96 6C
        C0 7A 3D 46 45 27 2E 31 FF 38 50 81 39 EB 0A 71"""

        btv = txt2bin(tv)
        res = K12.new().read(64)
        self.assertEqual(res, btv)

    def test_zero_3(self):
        tv = """E8 DC 56 36 42 F7 22 8C 84 68 4C 89 84 05 D3 A8
        34 79 91 58 C0 79 B1 28 80 27 7A 1D 28 E2 FF 6D"""

        btv = txt2bin(tv)
        res = K12.new().read(10032)
        self.assertEqual(res[-32:], btv)

    def test_ptn_1(self):
        tv = """2B DA 92 45 0E 8B 14 7F 8A 7C B6 29 E7 84 A0 58
        EF CA 7C F7 D8 21 8E 02 D3 45 DF AA 65 24 4A 1F"""

        btv = txt2bin(tv)
        res = K12.new(data=ptn(1)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17(self):
        tv = """6B F7 5F A2 23 91 98 DB 47 72 E3 64 78 F8 E1 9B
        0F 37 12 05 F6 A9 A9 3A 27 3F 51 DF 37 12 28 88"""

        btv = txt2bin(tv)
        res = K12.new(data=ptn(17)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_2(self):
        tv = """0C 31 5E BC DE DB F6 14 26 DE 7D CF 8F B7 25 D1
        E7 46 75 D7 F5 32 7A 50 67 F3 67 B1 08 EC B6 7C"""

        btv = txt2bin(tv)
        res = K12.new(data=ptn(17**2)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_3(self):
        tv = """CB 55 2E 2E C7 7D 99 10 70 1D 57 8B 45 7D DF 77
        2C 12 E3 22 E4 EE 7F E4 17 F9 2C 75 8F 0D 59 D0"""

        btv = txt2bin(tv)
        res = K12.new(data=ptn(17**3)).read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_4(self):
        tv = """87 01 04 5E 22 20 53 45 FF 4D DA 05 55 5C BB 5C
        3A F1 A7 71 C2 B8 9B AE F3 7D B4 3D 99 98 B9 FE"""

        btv = txt2bin(tv)
        data = ptn(17**4)

        # All at once
        res = K12.new(data=data).read(32)
        self.assertEqual(res, btv)

        # Byte by byte
        k12 = K12.new()
        for x in data:
            k12.update(bchr(x))
        res = k12.read(32)
        self.assertEqual(res, btv)

        # Chunks of various prime sizes
        for chunk_size in (13, 17, 19, 23, 31):
            k12 = K12.new()
            for x in chunked(data, chunk_size):
                k12.update(x)
            res = k12.read(32)
            self.assertEqual(res, btv)

    def test_ptn_17_5(self):
        tv = """84 4D 61 09 33 B1 B9 96 3C BD EB 5A E3 B6 B0 5C
        C7 CB D6 7C EE DF 88 3E B6 78 A0 A8 E0 37 16 82"""

        btv = txt2bin(tv)
        data = ptn(17**5)

        # All at once
        res = K12.new(data=data).read(32)
        self.assertEqual(res, btv)

        # Chunks
        k12 = K12.new()
        for chunk in chunked(data, 8192):
            k12.update(chunk)
        res = k12.read(32)
        self.assertEqual(res, btv)

    def test_ptn_17_6(self):
        tv = """3C 39 07 82 A8 A4 E8 9F A6 36 7F 72 FE AA F1 32
        55 C8 D9 58 78 48 1D 3C D8 CE 85 F5 8E 88 0A F8"""

        btv = txt2bin(tv)
        data = ptn(17**6)

        # All at once
        res = K12.new(data=data).read(32)
        self.assertEqual(res, btv)

    def test_ptn_c_1(self):
        tv = """FA B6 58 DB 63 E9 4A 24 61 88 BF 7A F6 9A 13 30
        45 F4 6E E9 84 C5 6E 3C 33 28 CA AF 1A A1 A5 83"""

        btv = txt2bin(tv)
        custom = ptn(1)

        # All at once
        res = K12.new(custom=custom).read(32)
        self.assertEqual(res, btv)

    def test_ptn_c_41(self):
        tv = """D8 48 C5 06 8C ED 73 6F 44 62 15 9B 98 67 FD 4C
        20 B8 08 AC C3 D5 BC 48 E0 B0 6B A0 A3 76 2E C4"""

        btv = txt2bin(tv)
        custom = ptn(41)

        # All at once
        res = K12.new(data=b'\xFF', custom=custom).read(32)
        self.assertEqual(res, btv)

    def test_ptn_c_41_2(self):
        tv = """C3 89 E5 00 9A E5 71 20 85 4C 2E 8C 64 67 0A C0
        13 58 CF 4C 1B AF 89 44 7A 72 42 34 DC 7C ED 74"""

        btv = txt2bin(tv)
        custom = ptn(41**2)

        # All at once
        res = K12.new(data=b'\xFF' * 3, custom=custom).read(32)
        self.assertEqual(res, btv)

    def test_ptn_c_41_3(self):
        tv = """75 D2 F8 6A 2E 64 45 66 72 6B 4F BC FC 56 57 B9
        DB CF 07 0C 7B 0D CA 06 45 0A B2 91 D7 44 3B CF"""

        btv = txt2bin(tv)
        custom = ptn(41**3)

        # All at once
        res = K12.new(data=b'\xFF' * 7, custom=custom).read(32)
        self.assertEqual(res, btv)

    ###

    def test_1(self):
        tv = "fd608f91d81904a9916e78a18f65c157a78d63f93d8f6367db0524526a5ea2bb"

        btv = txt2bin(tv)
        res = K12.new(data=b'', custom=ptn(100)).read(32)
        self.assertEqual(res, btv)

    def test_2(self):
        tv4 = "5a4ec9a649f81916d4ce1553492962f7868abf8dd1ceb2f0cb3682ea95cda6a6"
        tv3 = "441688fe4fe4ae9425eb3105eb445eb2b3a6f67b66eff8e74ebfbc49371f6d4c"
        tv2 = "17269a57759af0214c84a0fd9bc851f4d95f80554cfed4e7da8a6ee1ff080131"
        tv1 = "33826990c09dc712ba7224f0d9be319e2720de95a4c1afbd2211507dae1c703a"
        tv0 = "9f4d3aba908ddc096e4d3a71da954f917b9752f05052b9d26d916a6fbc75bf3e"

        res = K12.new(data=b'A' * (8192 - 4), custom=b'B').read(32)
        self.assertEqual(res, txt2bin(tv4))

        res = K12.new(data=b'A' * (8192 - 3), custom=b'B').read(32)
        self.assertEqual(res, txt2bin(tv3))

        res = K12.new(data=b'A' * (8192 - 2), custom=b'B').read(32)
        self.assertEqual(res, txt2bin(tv2))

        res = K12.new(data=b'A' * (8192 - 1), custom=b'B').read(32)
        self.assertEqual(res, txt2bin(tv1))

        res = K12.new(data=b'A' * (8192 - 0), custom=b'B').read(32)
        self.assertEqual(res, txt2bin(tv0))


def get_tests(config={}):
    tests = []
    tests += list_test_cases(KangarooTwelveTest)
    tests += list_test_cases(KangarooTwelveTV)
    return tests


if __name__ == '__main__':
    def suite():
        return unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
