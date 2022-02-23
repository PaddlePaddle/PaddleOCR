# -*- coding: utf-8 -*-
#
#  SelfTest/Protocol/test_KDF.py: Self-test for key derivation functions
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

import unittest
from binascii import unhexlify

from Crypto.Util.py3compat import b, bchr

from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors_wycheproof
from Crypto.Hash import SHA1, HMAC, SHA256, MD5, SHA224, SHA384, SHA512
from Crypto.Cipher import AES, DES3

from Crypto.Protocol.KDF import (PBKDF1, PBKDF2, _S2V, HKDF, scrypt,
                                 bcrypt, bcrypt_check)

from Crypto.Protocol.KDF import _bcrypt_decode


def t2b(t):
    if t is None:
        return None
    t2 = t.replace(" ", "").replace("\n", "")
    return unhexlify(b(t2))


class TestVector(object):
    pass


class PBKDF1_Tests(unittest.TestCase):

    # List of tuples with test data.
    # Each tuple is made up by:
    #       Item #0: a pass phrase
    #       Item #1: salt (8 bytes encoded in hex)
    #       Item #2: output key length
    #       Item #3: iterations to use
    #       Item #4: expected result (encoded in hex)
    _testData = (
            # From http://www.di-mgt.com.au/cryptoKDFs.html#examplespbkdf
            ("password", "78578E5A5D63CB06", 16, 1000, "DC19847E05C64D2FAF10EBFB4A3D2A20"),
    )

    def test1(self):
        v = self._testData[0]
        res = PBKDF1(v[0], t2b(v[1]), v[2], v[3], SHA1)
        self.assertEqual(res, t2b(v[4]))


class PBKDF2_Tests(unittest.TestCase):

    # List of tuples with test data.
    # Each tuple is made up by:
    #       Item #0: a pass phrase
    #       Item #1: salt (encoded in hex)
    #       Item #2: output key length
    #       Item #3: iterations to use
    #       Item #4: hash module
    #       Item #5: expected result (encoded in hex)
    _testData = (
            # From http://www.di-mgt.com.au/cryptoKDFs.html#examplespbkdf
            ("password","78578E5A5D63CB06",24,2048,     SHA1, "BFDE6BE94DF7E11DD409BCE20A0255EC327CB936FFE93643"),
            # From RFC 6050
            ("password","73616c74", 20, 1,              SHA1, "0c60c80f961f0e71f3a9b524af6012062fe037a6"),
            ("password","73616c74", 20, 2,              SHA1, "ea6c014dc72d6f8ccd1ed92ace1d41f0d8de8957"),
            ("password","73616c74", 20, 4096,           SHA1, "4b007901b765489abead49d926f721d065a429c1"),
            ("passwordPASSWORDpassword","73616c7453414c5473616c7453414c5473616c7453414c5473616c7453414c5473616c74",
                                    25, 4096,           SHA1, "3d2eec4fe41c849b80c8d83662c0e44a8b291a964cf2f07038"),
            ( 'pass\x00word',"7361006c74",16,4096,      SHA1, "56fa6aa75548099dcc37d7f03425e0c3"),
            # From draft-josefsson-scrypt-kdf-01, Chapter 10
            ( 'passwd', '73616c74', 64, 1,              SHA256, "55ac046e56e3089fec1691c22544b605f94185216dde0465e68b9d57c20dacbc49ca9cccf179b645991664b39d77ef317c71b845b1e30bd509112041d3a19783"),
            ( 'Password', '4e61436c', 64, 80000,        SHA256, "4ddcd8f60b98be21830cee5ef22701f9641a4418d04c0414aeff08876b34ab56a1d425a1225833549adb841b51c9b3176a272bdebba1d078478f62b397f33c8d"),
        )

    def test1(self):
        # Test only for HMAC-SHA1 as PRF

        def prf_SHA1(p,s):
            return HMAC.new(p,s,SHA1).digest()

        def prf_SHA256(p,s):
            return HMAC.new(p,s,SHA256).digest()

        for i in range(len(self._testData)):
            v = self._testData[i]
            password = v[0]
            salt = t2b(v[1])
            out_len = v[2]
            iters = v[3]
            hash_mod = v[4]
            expected = t2b(v[5])

            if hash_mod is SHA1:
                res = PBKDF2(password, salt, out_len, iters)
                self.assertEqual(res, expected)

                res = PBKDF2(password, salt, out_len, iters, prf_SHA1)
                self.assertEqual(res, expected)
            else:
                res = PBKDF2(password, salt, out_len, iters, prf_SHA256)
                self.assertEqual(res, expected)

    def test2(self):
        # Verify that prf and hmac_hash_module are mutual exclusive
        def prf_SHA1(p,s):
            return HMAC.new(p,s,SHA1).digest()

        self.assertRaises(ValueError, PBKDF2, b("xxx"), b("yyy"), 16, 100,
                          prf=prf_SHA1, hmac_hash_module=SHA1)

    def test3(self):
        # Verify that hmac_hash_module works like prf

        password = b("xxx")
        salt = b("yyy")

        for hashmod in (MD5, SHA1, SHA224, SHA256, SHA384, SHA512):

            pr1 = PBKDF2(password, salt, 16, 100,
                         prf=lambda p, s: HMAC.new(p,s,hashmod).digest())
            pr2 = PBKDF2(password, salt, 16, 100, hmac_hash_module=hashmod)

            self.assertEqual(pr1, pr2)

    def test4(self):
        # Verify that PBKDF2 can take bytes or strings as password or salt
        k1 = PBKDF2("xxx", b("yyy"), 16, 10)
        k2 = PBKDF2(b("xxx"), b("yyy"), 16, 10)
        self.assertEqual(k1, k2)

        k1 = PBKDF2(b("xxx"), "yyy", 16, 10)
        k2 = PBKDF2(b("xxx"), b("yyy"), 16, 10)
        self.assertEqual(k1, k2)


class S2V_Tests(unittest.TestCase):

    # Sequence of test vectors.
    # Each test vector is made up by:
    #   Item #0: a tuple of strings
    #   Item #1: an AES key
    #   Item #2: the result
    #   Item #3: the cipher module S2V is based on
    # Everything is hex encoded
    _testData = [

            # RFC5297, A.1
            (
             (  '101112131415161718191a1b1c1d1e1f2021222324252627',
                '112233445566778899aabbccddee' ),
            'fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0',
            '85632d07c6e8f37f950acd320a2ecc93',
            AES
            ),

            # RFC5297, A.2
            (
             (  '00112233445566778899aabbccddeeffdeaddadadeaddadaffeeddcc'+
                'bbaa99887766554433221100',
                '102030405060708090a0',
                '09f911029d74e35bd84156c5635688c0',
                '7468697320697320736f6d6520706c61'+
                '696e7465787420746f20656e63727970'+
                '74207573696e67205349562d414553'),
            '7f7e7d7c7b7a79787776757473727170',
            '7bdb6e3b432667eb06f4d14bff2fbd0f',
            AES
            ),

        ]

    def test1(self):
        """Verify correctness of test vector"""
        for tv in self._testData:
            s2v = _S2V.new(t2b(tv[1]), tv[3])
            for s in tv[0]:
                s2v.update(t2b(s))
            result = s2v.derive()
            self.assertEqual(result, t2b(tv[2]))

    def test2(self):
        """Verify that no more than 127(AES) and 63(TDES)
        components are accepted."""
        key = bchr(0) * 8 + bchr(255) * 8
        for module in (AES, DES3):
            s2v = _S2V.new(key, module)
            max_comps = module.block_size*8-1
            for i in range(max_comps):
                s2v.update(b("XX"))
            self.assertRaises(TypeError, s2v.update, b("YY"))


class HKDF_Tests(unittest.TestCase):

    # Test vectors from RFC5869, Appendix A
    # Each tuple is made up by:
    #       Item #0: hash module
    #       Item #1: secret
    #       Item #2: salt
    #       Item #3: context
    #       Item #4: expected result
    _test_vector  = (
            (
                SHA256,
                "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                "000102030405060708090a0b0c",
                "f0f1f2f3f4f5f6f7f8f9",
                42,
                "3cb25f25faacd57a90434f64d0362f2a" +
                "2d2d0a90cf1a5a4c5db02d56ecc4c5bf" +
                "34007208d5b887185865"
            ),
            (
                SHA256,
                "000102030405060708090a0b0c0d0e0f" +
                "101112131415161718191a1b1c1d1e1f" +
                "202122232425262728292a2b2c2d2e2f" +
                "303132333435363738393a3b3c3d3e3f" +
                "404142434445464748494a4b4c4d4e4f",
                "606162636465666768696a6b6c6d6e6f" +
                "707172737475767778797a7b7c7d7e7f" +
                "808182838485868788898a8b8c8d8e8f" +
                "909192939495969798999a9b9c9d9e9f" +
                "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
                "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf" +
                "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf" +
                "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf" +
                "e0e1e2e3e4e5e6e7e8e9eaebecedeeef" +
                "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
                82,
                "b11e398dc80327a1c8e7f78c596a4934" +
                "4f012eda2d4efad8a050cc4c19afa97c" +
                "59045a99cac7827271cb41c65e590e09" +
                "da3275600c2f09b8367793a9aca3db71" +
                "cc30c58179ec3e87c14c01d5c1f3434f" +
                "1d87"
            ),
            (
                SHA256,
                "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                None,
                None,
                42,
                "8da4e775a563c18f715f802a063c5a31" +
                "b8a11f5c5ee1879ec3454e5f3c738d2d" +
                "9d201395faa4b61a96c8"
            ),
            (
                SHA1,
                "0b0b0b0b0b0b0b0b0b0b0b",
                "000102030405060708090a0b0c",
                "f0f1f2f3f4f5f6f7f8f9",
                42,
                "085a01ea1b10f36933068b56efa5ad81" +
                "a4f14b822f5b091568a9cdd4f155fda2" +
                "c22e422478d305f3f896"
            ),
            (
                SHA1,
                "000102030405060708090a0b0c0d0e0f" +
                "101112131415161718191a1b1c1d1e1f" +
                "202122232425262728292a2b2c2d2e2f" +
                "303132333435363738393a3b3c3d3e3f" +
                "404142434445464748494a4b4c4d4e4f",
                "606162636465666768696a6b6c6d6e6f" +
                "707172737475767778797a7b7c7d7e7f" +
                "808182838485868788898a8b8c8d8e8f" +
                "909192939495969798999a9b9c9d9e9f" +
                "a0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
                "b0b1b2b3b4b5b6b7b8b9babbbcbdbebf" +
                "c0c1c2c3c4c5c6c7c8c9cacbcccdcecf" +
                "d0d1d2d3d4d5d6d7d8d9dadbdcdddedf" +
                "e0e1e2e3e4e5e6e7e8e9eaebecedeeef" +
                "f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
                82,
                "0bd770a74d1160f7c9f12cd5912a06eb" +
                "ff6adcae899d92191fe4305673ba2ffe" +
                "8fa3f1a4e5ad79f3f334b3b202b2173c" +
                "486ea37ce3d397ed034c7f9dfeb15c5e" +
                "927336d0441f4c4300e2cff0d0900b52" +
                "d3b4"
            ),
            (
                SHA1,
                "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b",
                "",
                "",
                42,
                "0ac1af7002b3d761d1e55298da9d0506" +
                "b9ae52057220a306e07b6b87e8df21d0" +
                "ea00033de03984d34918"
            ),
            (
                SHA1,
                "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c",
                None,
                "",
                42,
                "2c91117204d745f3500d636a62f64f0a" +
                "b3bae548aa53d423b0d1f27ebba6f5e5" +
                "673a081d70cce7acfc48"
            )
        )

    def test1(self):
        for tv in self._test_vector:
            secret, salt, info, exp = [ t2b(tv[x]) for x in (1,2,3,5) ]
            key_len, hashmod = [ tv[x] for x in (4,0) ]

            output = HKDF(secret, key_len, salt, hashmod, 1, info)
            self.assertEqual(output, exp)

    def test2(self):
        ref = HKDF(b("XXXXXX"), 12, b("YYYY"), SHA1)

        # Same output, but this time split over 2 keys
        key1, key2 = HKDF(b("XXXXXX"), 6, b("YYYY"), SHA1, 2)
        self.assertEqual((ref[:6], ref[6:]), (key1, key2))

        # Same output, but this time split over 3 keys
        key1, key2, key3 = HKDF(b("XXXXXX"), 4, b("YYYY"), SHA1, 3)
        self.assertEqual((ref[:4], ref[4:8], ref[8:]), (key1, key2, key3))


class scrypt_Tests(unittest.TestCase):

    # Test vectors taken from
    # https://tools.ietf.org/html/rfc7914
    # - password
    # - salt
    # - N
    # - r
    # - p
    data = (
                (
                    "",
                    "",
                    16,     # 2K
                    1,
                    1,
                    """
                    77 d6 57 62 38 65 7b 20 3b 19 ca 42 c1 8a 04 97
                    f1 6b 48 44 e3 07 4a e8 df df fa 3f ed e2 14 42
                    fc d0 06 9d ed 09 48 f8 32 6a 75 3a 0f c8 1f 17
                    e8 d3 e0 fb 2e 0d 36 28 cf 35 e2 0c 38 d1 89 06
                    """
                ),
                (
                    "password",
                    "NaCl",
                    1024,   # 1M
                    8,
                    16,
                    """
                    fd ba be 1c 9d 34 72 00 78 56 e7 19 0d 01 e9 fe
                    7c 6a d7 cb c8 23 78 30 e7 73 76 63 4b 37 31 62
                    2e af 30 d9 2e 22 a3 88 6f f1 09 27 9d 98 30 da
                    c7 27 af b9 4a 83 ee 6d 83 60 cb df a2 cc 06 40
                    """
                ),
                (
                    "pleaseletmein",
                    "SodiumChloride",
                    16384,  # 16M
                    8,
                    1,
                    """
                    70 23 bd cb 3a fd 73 48 46 1c 06 cd 81 fd 38 eb
                    fd a8 fb ba 90 4f 8e 3e a9 b5 43 f6 54 5d a1 f2
                    d5 43 29 55 61 3f 0f cf 62 d4 97 05 24 2a 9a f9
                    e6 1e 85 dc 0d 65 1e 40 df cf 01 7b 45 57 58 87
                    """
                ),
                (
                    "pleaseletmein",
                    "SodiumChloride",
                    1048576, # 1G
                    8,
                    1,
                    """
                    21 01 cb 9b 6a 51 1a ae ad db be 09 cf 70 f8 81
                    ec 56 8d 57 4a 2f fd 4d ab e5 ee 98 20 ad aa 47
                    8e 56 fd 8f 4b a5 d0 9f fa 1c 6d 92 7c 40 f4 c3
                    37 30 40 49 e8 a9 52 fb cb f4 5c 6f a7 7a 41 a4
                    """
                ),
            )

    def setUp(self):
        new_test_vectors = []
        for tv in self.data:
            new_tv = TestVector()
            new_tv.P = b(tv[0])
            new_tv.S = b(tv[1])
            new_tv.N = tv[2]
            new_tv.r = tv[3]
            new_tv.p = tv[4]
            new_tv.output = t2b(tv[5])
            new_tv.dkLen = len(new_tv.output)
            new_test_vectors.append(new_tv)
        self.data = new_test_vectors

    def test2(self):

        for tv in self.data:
            try:
                output = scrypt(tv.P, tv.S, tv.dkLen, tv.N, tv.r, tv.p)
            except ValueError as e:
                if " 2 " in str(e) and tv.N >= 1048576:
                    import warnings
                    warnings.warn("Not enough memory to unit test scrypt() with N=1048576", RuntimeWarning)
                    continue
                else:
                    raise e
            self.assertEqual(output, tv.output)

    def test3(self):
        ref = scrypt(b("password"), b("salt"), 12, 16, 1, 1)

        # Same output, but this time split over 2 keys
        key1, key2 = scrypt(b("password"), b("salt"), 6, 16, 1, 1, 2)
        self.assertEqual((ref[:6], ref[6:]), (key1, key2))

        # Same output, but this time split over 3 keys
        key1, key2, key3 = scrypt(b("password"), b("salt"), 4, 16, 1, 1, 3)
        self.assertEqual((ref[:4], ref[4:8], ref[8:]), (key1, key2, key3))


class bcrypt_Tests(unittest.TestCase):

    def test_negative_cases(self):
        self.assertRaises(ValueError, bcrypt, b"1" * 73, 10)
        self.assertRaises(ValueError, bcrypt, b"1" * 10, 3)
        self.assertRaises(ValueError, bcrypt, b"1" * 10, 32)
        self.assertRaises(ValueError, bcrypt, b"1" * 10, 4, salt=b"")
        self.assertRaises(ValueError, bcrypt, b"1" * 10, 4, salt=b"1")
        self.assertRaises(ValueError, bcrypt, b"1" * 10, 4, salt=b"1" * 17)
        self.assertRaises(ValueError, bcrypt, b"1\x00" * 10, 4)

    def test_bytearray_mismatch(self):
        ref = bcrypt("pwd", 4)
        bcrypt_check("pwd", ref)
        bref = bytearray(ref)
        bcrypt_check("pwd", bref)

        wrong = ref[:-1] + bchr(bref[-1] ^ 0x01)
        self.assertRaises(ValueError, bcrypt_check, "pwd", wrong)

        wrong = b"x" + ref[1:]
        self.assertRaises(ValueError, bcrypt_check, "pwd", wrong)

    # https://github.com/patrickfav/bcrypt/wiki/Published-Test-Vectors

    def test_empty_password(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"", 4, b"zVHmKQtGGQob.b/Nc7l9NO", b"$2a$04$zVHmKQtGGQob.b/Nc7l9NO8UlrYcW05FiuCj/SxsFO/ZtiN9.mNzy"),
            (b"", 5, b"zVHmKQtGGQob.b/Nc7l9NO", b"$2a$05$zVHmKQtGGQob.b/Nc7l9NOWES.1hkVBgy5IWImh9DOjKNU8atY4Iy"),
            (b"", 6, b"zVHmKQtGGQob.b/Nc7l9NO", b"$2a$06$zVHmKQtGGQob.b/Nc7l9NOjOl7l4oz3WSh5fJ6414Uw8IXRAUoiaO"),
            (b"", 7, b"zVHmKQtGGQob.b/Nc7l9NO", b"$2a$07$zVHmKQtGGQob.b/Nc7l9NOBsj1dQpBA1HYNGpIETIByoNX9jc.hOi"),
            (b"", 8, b"zVHmKQtGGQob.b/Nc7l9NO", b"$2a$08$zVHmKQtGGQob.b/Nc7l9NOiLTUh/9MDpX86/DLyEzyiFjqjBFePgO"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_random_password_and_salt_short_pw(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"<.S.2K(Zq'", 4, b"VYAclAMpaXY/oqAo9yUpku", b"$2a$04$VYAclAMpaXY/oqAo9yUpkuWmoYywaPzyhu56HxXpVltnBIfmO9tgu"),
            (b"5.rApO%5jA", 5, b"kVNDrnYKvbNr5AIcxNzeIu", b"$2a$05$kVNDrnYKvbNr5AIcxNzeIuRcyIF5cZk6UrwHGxENbxP5dVv.WQM/G"),
            (b"oW++kSrQW^", 6, b"QLKkRMH9Am6irtPeSKN5sO", b"$2a$06$QLKkRMH9Am6irtPeSKN5sObJGr3j47cO6Pdf5JZ0AsJXuze0IbsNm"),
            (b"ggJ\\KbTnDG", 7, b"4H896R09bzjhapgCPS/LYu", b"$2a$07$4H896R09bzjhapgCPS/LYuMzAQluVgR5iu/ALF8L8Aln6lzzYXwbq"),
            (b"49b0:;VkH/", 8, b"hfvO2retKrSrx5f2RXikWe", b"$2a$08$hfvO2retKrSrx5f2RXikWeFWdtSesPlbj08t/uXxCeZoHRWDz/xFe"),
            (b">9N^5jc##'", 9, b"XZLvl7rMB3EvM0c1.JHivu", b"$2a$09$XZLvl7rMB3EvM0c1.JHivuIDPJWeNJPTVrpjZIEVRYYB/mF6cYgJK"),
            (b"\\$ch)s4WXp", 10, b"aIjpMOLK5qiS9zjhcHR5TO", b"$2a$10$aIjpMOLK5qiS9zjhcHR5TOU7v2NFDmcsBmSFDt5EHOgp/jeTF3O/q"),
            (b"RYoj\\_>2P7", 12, b"esIAHiQAJNNBrsr5V13l7.", b"$2a$12$esIAHiQAJNNBrsr5V13l7.RFWWJI2BZFtQlkFyiWXjou05GyuREZa"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_random_password_and_salt_long_pw(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"^Q&\"]A`%/A(BVGt>QaX0M-#<Q148&f", 4, b"vrRP5vQxyD4LrqiLd/oWRO", b"$2a$04$vrRP5vQxyD4LrqiLd/oWROgrrGINsw3gb4Ga5x2sn01jNmiLVECl6"),
            (b"nZa!rRf\\U;OL;R?>1ghq_+\":Y0CRmY", 5, b"YuQvhokOGVnevctykUYpKu", b"$2a$05$YuQvhokOGVnevctykUYpKutZD2pWeGGYn3auyLOasguMY3/0BbIyq"),
            (b"F%uN/j>[GuB7-jB'_Yj!Tnb7Y!u^6)", 6, b"5L3vpQ0tG9O7k5gQ8nAHAe", b"$2a$06$5L3vpQ0tG9O7k5gQ8nAHAe9xxQiOcOLh8LGcI0PLWhIznsDt.S.C6"),
            (b"Z>BobP32ub\"Cfe*Q<<WUq3rc=[GJr-", 7, b"hp8IdLueqE6qFh1zYycUZ.", b"$2a$07$hp8IdLueqE6qFh1zYycUZ.twmUH8eSTPQAEpdNXKMlwms9XfKqfea"),
            (b"Ik&8N['7*[1aCc1lOm8\\jWeD*H$eZM", 8, b"2ANDTYCB9m7vf0Prh7rSru", b"$2a$08$2ANDTYCB9m7vf0Prh7rSrupqpO3jJOkIz2oW/QHB4lCmK7qMytGV6"),
            (b"O)=%3[E$*q+>-q-=tRSjOBh8\\mLNW.", 9, b"nArqOfdCsD9kIbVnAixnwe", b"$2a$09$nArqOfdCsD9kIbVnAixnwe6s8QvyPYWtQBpEXKir2OJF9/oNBsEFe"),
            (b"/MH51`!BP&0tj3%YCA;Xk%e3S`o\\EI", 10, b"ePiAc.s.yoBi3B6p1iQUCe", b"$2a$10$ePiAc.s.yoBi3B6p1iQUCezn3mraLwpVJ5XGelVyYFKyp5FZn/y.u"),
            (b"ptAP\"mcg6oH.\";c0U2_oll.OKi<!ku", 12, b"aroG/pwwPj1tU5fl9a9pkO", b"$2a$12$aroG/pwwPj1tU5fl9a9pkO4rydAmkXRj/LqfHZOSnR6LGAZ.z.jwa"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_same_password_and_random_salt(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"Q/A:k3DP;X@=<0\"hg&9c", 4, b"wbgDTvLMtyjQlNK7fjqwyO", b"$2a$04$wbgDTvLMtyjQlNK7fjqwyOakBoACQuYh11.VsKNarF4xUIOBWgD6S"),
            (b"Q/A:k3DP;X@=<0\"hg&9c", 5, b"zbAaOmloOhxiKItjznRqru", b"$2a$05$zbAaOmloOhxiKItjznRqrunRqHlu3MAa7pMGv26Rr3WwyfGcwoRm6"),
            (b"Q/A:k3DP;X@=<0\"hg&9c", 6, b"aOK0bWUvLI0qLkc3ti5jyu", b"$2a$06$aOK0bWUvLI0qLkc3ti5jyuAIQoqRzuqoK09kQqQ6Ou/YKDhW50/qa"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_same_password_and_salt_increasing_cost_factor(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"o<&+X'F4AQ8H,LU,N`&r", 4, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$04$BK5u.QHk1Driey7bvnFTH.3smGwxd91PtoK2GxH5nZ7pcBsYX4lMq"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 5, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$05$BK5u.QHk1Driey7bvnFTH.t5P.jZvFBMzDB1IY4PwkkRPOyVbEtFG"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 6, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$06$BK5u.QHk1Driey7bvnFTH.6Ea1Z5db2p25CPXZbxb/3OyKQagg3pa"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 7, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$07$BK5u.QHk1Driey7bvnFTH.sruuQi8Lhv/0LWKDvNp3AGFk7ltdkm6"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 8, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$08$BK5u.QHk1Driey7bvnFTH.IE7KsaUzc4m7gzAMlyUPUeiYyACWe0q"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 9, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$09$BK5u.QHk1Driey7bvnFTH.1v4Xj1dwkp44QNg0cVAoQt4FQMMrvnS"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 10, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$10$BK5u.QHk1Driey7bvnFTH.ESINe9YntUMcVgFDfkC.Vbhc9vMhNX2"),
            (b"o<&+X'F4AQ8H,LU,N`&r", 12, b"BK5u.QHk1Driey7bvnFTH.", b"$2a$12$BK5u.QHk1Driey7bvnFTH.QM1/nnGe/f5cTzb6XTTi/vMzcAnycqG"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_long_passwords(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"g*3Q45=\"8NNgpT&mbMJ$Omfr.#ZeW?FP=CE$#roHd?97uL0F-]`?u73c\"\\[.\"*)qU34@VG",
             4, b"T2XJ5MOWvHQZRijl8LIKkO", b"$2a$04$T2XJ5MOWvHQZRijl8LIKkOQKIyX75KBfuLsuRYOJz5OjwBNF2lM8a"),
            (b"\\M+*8;&QE=Ll[>5?Ui\"^ai#iQH7ZFtNMfs3AROnIncE9\"BNNoEgO[[*Yk8;RQ(#S,;I+aT",
             5, b"wgkOlGNXIVE2fWkT3gyRoO", b"$2a$05$wgkOlGNXIVE2fWkT3gyRoOqWi4gbi1Wv2Q2Jx3xVs3apl1w.Wtj8C"),
            (b"M.E1=dt<.L0Q&p;94NfGm_Oo23+Kpl@M5?WIAL.[@/:'S)W96G8N^AWb7_smmC]>7#fGoB",
             6, b"W9zTCl35nEvUukhhFzkKMe", b"$2a$06$W9zTCl35nEvUukhhFzkKMekjT9/pj7M0lihRVEZrX3m8/SBNZRX7i"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_increasing_password_length(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            (b"a", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.l4WvgHIVg17ZawDIrDM2IjlE64GDNQS"),
            (b"aa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.AyUxBk.ThHlsLvRTH7IqcG7yVHJ3SXq"),
            (b"aaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.BxOVac5xPB6XFdRc/ZrzM9FgZkqmvbW"),
            (b"aaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.Qbr209bpCtfl5hN7UQlG/L4xiD3AKau"),
            (b"aaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.oWszihPjDZI0ypReKsaDOW1jBl7oOii"),
            (b"aaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ./k.Xxn9YiqtV/sxh3EHbnOHd0Qsq27K"),
            (b"aaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.PYJqRFQbgRbIjMd5VNKmdKS4sBVOyDe"),
            (b"aaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ..VMYfzaw1wP/SGxowpLeGf13fxCCt.q"),
            (b"aaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.5B0p054nO5WgAD1n04XslDY/bqY9RJi"),
            (b"aaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.INBTgqm7sdlBJDg.J5mLMSRK25ri04y"),
            (b"aaaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.s3y7CdFD0OR5p6rsZw/eZ.Dla40KLfm"),
            (b"aaaaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.Jx742Djra6Q7PqJWnTAS.85c28g.Siq"),
            (b"aaaaaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.oKMXW3EZcPHcUV0ib5vDBnh9HojXnLu"),
            (b"aaaaaaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.w6nIjWpDPNSH5pZUvLjC1q25ONEQpeS"),
            (b"aaaaaaaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.k1b2/r9A/hxdwKEKurg6OCn4MwMdiGq"),
            (b"aaaaaaaaaaaaaaaa", 4, b"5DCebwootqWMCp59ISrMJ.", b"$2a$04$5DCebwootqWMCp59ISrMJ.3prCNHVX1Ws.7Hm2bJxFUnQOX9f7DFa"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_non_ascii_characters(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            ("àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝðÐ", 4, b"D3qS2aoTVyqM7z8v8crLm.", b"$2a$04$D3qS2aoTVyqM7z8v8crLm.3nKt4CzBZJbyFB.ZebmfCvRw7BGs.Xm"),
            ("àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝðÐ", 5, b"VA1FujiOCMPkUHQ8kF7IaO", b"$2a$05$VA1FujiOCMPkUHQ8kF7IaOg7NGaNvpxwWzSluQutxEVmbZItRTsAa"),
            ("àèìòùÀÈÌÒÙáéíóúýÁÉÍÓÚÝðÐ", 6, b"TXiaNrPeBSz5ugiQlehRt.", b"$2a$06$TXiaNrPeBSz5ugiQlehRt.gwpeDQnXWteQL4z2FulouBr6G7D9KUi"),
            ("âêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿ", 4, b"YTn1Qlvps8e1odqMn6G5x.", b"$2a$04$YTn1Qlvps8e1odqMn6G5x.85pqKql6w773EZJAExk7/BatYAI4tyO"),
            ("âêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿ", 5, b"C.8k5vJKD2NtfrRI9o17DO", b"$2a$05$C.8k5vJKD2NtfrRI9o17DOfIW0XnwItA529vJnh2jzYTb1QdoY0py"),
            ("âêîôûÂÊÎÔÛãñõÃÑÕäëïöüÿ", 6, b"xqfRPj3RYAgwurrhcA6uRO", b"$2a$06$xqfRPj3RYAgwurrhcA6uROtGlXDp/U6/gkoDYHwlubtcVcNft5.vW"),
            ("ÄËÏÖÜŸåÅæÆœŒßçÇøØ¢¿¡€", 4, b"y8vGgMmr9EdyxP9rmMKjH.", b"$2a$04$y8vGgMmr9EdyxP9rmMKjH.wv2y3r7yRD79gykQtmb3N3zrwjKsyay"),
            ("ÄËÏÖÜŸåÅæÆœŒßçÇøØ¢¿¡€", 5, b"iYH4XIKAOOm/xPQs7xKP1u", b"$2a$05$iYH4XIKAOOm/xPQs7xKP1upD0cWyMn3Jf0ZWiizXbEkVpS41K1dcO"),
            ("ÄËÏÖÜŸåÅæÆœŒßçÇøØ¢¿¡€", 6, b"wCOob.D0VV8twafNDB2ape", b"$2a$06$wCOob.D0VV8twafNDB2apegiGD5nqF6Y1e6K95q6Y.R8C4QGd265q"),
            ("ΔημοσιεύθηκεστηνΕφημερίδατης", 4, b"E5SQtS6P4568MDXW7cyUp.", b"$2a$04$E5SQtS6P4568MDXW7cyUp.18wfDisKZBxifnPZjAI1d/KTYMfHPYO"),
            ("АБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмН", 4, b"03e26gQFHhQwRNf81/ww9.", b"$2a$04$03e26gQFHhQwRNf81/ww9.p1UbrNwxpzWjLuT.zpTLH4t/w5WhAhC"),
            ("нОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮю", 4, b"PHNoJwpXCfe32nUtLv2Upu", b"$2a$04$PHNoJwpXCfe32nUtLv2UpuhJXOzd4k7IdFwnEpYwfJVCZ/f/.8Pje"),
            ("電电電島岛島兔兔兎龜龟亀國国国區区区", 4, b"wU4/0i1TmNl2u.1jIwBX.u", b"$2a$04$wU4/0i1TmNl2u.1jIwBX.uZUaOL3Rc5ID7nlQRloQh6q5wwhV/zLW"),
            ("诶比伊艾弗豆贝尔维吾艾尺开艾丝维贼德", 4, b"P4kreGLhCd26d4WIy7DJXu", b"$2a$04$P4kreGLhCd26d4WIy7DJXusPkhxLvBouzV6OXkL5EB0jux0osjsry"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)

    def test_special_case_salt(self):
        # password, cost, salt, bcrypt hash
        tvs = [
            ("-O_=*N!2JP", 4, b"......................", b"$2a$04$......................JjuKLOX9OOwo5PceZZXSkaLDvdmgb82"),
            ("7B[$Q<4b>U", 5, b"......................", b"$2a$05$......................DRiedDQZRL3xq5A5FL8y7/6NM8a2Y5W"),
            (">d5-I_8^.h", 6, b"......................", b"$2a$06$......................5Mq1Ng8jgDY.uHNU4h5p/x6BedzNH2W"),
            (")V`/UM/]1t", 4, b".OC/.OC/.OC/.OC/.OC/.O", b"$2a$04$.OC/.OC/.OC/.OC/.OC/.OQIvKRDAam.Hm5/IaV/.hc7P8gwwIbmi"),
            (":@t2.bWuH]", 5, b".OC/.OC/.OC/.OC/.OC/.O", b"$2a$05$.OC/.OC/.OC/.OC/.OC/.ONDbUvdOchUiKmQORX6BlkPofa/QxW9e"),
            ("b(#KljF5s\"", 6, b".OC/.OC/.OC/.OC/.OC/.O", b"$2a$06$.OC/.OC/.OC/.OC/.OC/.OHfTd9e7svOu34vi1PCvOcAEq07ST7.K"),
            ("@3YaJ^Xs]*", 4, b"eGA.eGA.eGA.eGA.eGA.e.", b"$2a$04$eGA.eGA.eGA.eGA.eGA.e.stcmvh.R70m.0jbfSFVxlONdj1iws0C"),
            ("'\"5\\!k*C(p", 5, b"eGA.eGA.eGA.eGA.eGA.e.", b"$2a$05$eGA.eGA.eGA.eGA.eGA.e.vR37mVSbfdHwu.F0sNMvgn8oruQRghy"),
            ("edEu7C?$'W", 6, b"eGA.eGA.eGA.eGA.eGA.e.", b"$2a$06$eGA.eGA.eGA.eGA.eGA.e.tSq0FN8MWHQXJXNFnHTPQKtA.n2a..G"),
            ("N7dHmg\\PI^", 4, b"999999999999999999999u", b"$2a$04$999999999999999999999uCZfA/pLrlyngNDMq89r1uUk.bQ9icOu"),
            ("\"eJuHh!)7*", 5, b"999999999999999999999u", b"$2a$05$999999999999999999999uj8Pfx.ufrJFAoWFLjapYBS5vVEQQ/hK"),
            ("ZeDRJ:_tu:", 6, b"999999999999999999999u", b"$2a$06$999999999999999999999u6RB0P9UmbdbQgjoQFEJsrvrKe.BoU6q"),
        ]

        for (idx, (password, cost, salt64, result)) in enumerate(tvs):
            x = bcrypt(password, cost, salt=_bcrypt_decode(salt64))
            self.assertEqual(x, result)
            bcrypt_check(password, result)


class TestVectorsHKDFWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = "None"

    def add_tests(self, filename):

        def filter_algo(root):
            algo_name = root['algorithm']
            if algo_name == "HKDF-SHA-1":
                return SHA1
            elif algo_name == "HKDF-SHA-256":
                return SHA256
            elif algo_name == "HKDF-SHA-384":
                return SHA384
            elif algo_name == "HKDF-SHA-512":
                return SHA512
            else:
                raise ValueError("Unknown algorithm " + algo_name)

        def filter_size(unit):
            return int(unit['size'])

        result = load_test_vectors_wycheproof(("Protocol", "wycheproof"),
                                              filename,
                                              "Wycheproof HMAC (%s)" % filename,
                                              root_tag={'hash_module': filter_algo},
                                              unit_tag={'size': filter_size})
        return result

    def setUp(self):
        self.tv = []
        self.add_tests("hkdf_sha1_test.json")
        self.add_tests("hkdf_sha256_test.json")
        self.add_tests("hkdf_sha384_test.json")
        self.add_tests("hkdf_sha512_test.json")

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = "Wycheproof HKDF Test #%d (%s, %s)" % (tv.id, tv.comment, tv.filename)

        try:
            key = HKDF(tv.ikm, tv.size, tv.salt, tv.hash_module, 1, tv.info)
        except ValueError:
            assert not tv.valid
        else:
            if key != tv.okm:
                assert not tv.valid
            else:
                assert tv.valid
                self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)


def get_tests(config={}):
    wycheproof_warnings = config.get('wycheproof_warnings')

    if not config.get('slow_tests'):
        PBKDF2_Tests._testData = PBKDF2_Tests._testData[:3]
        scrypt_Tests.data = scrypt_Tests.data[:3]

    tests = []
    tests += list_test_cases(PBKDF1_Tests)
    tests += list_test_cases(PBKDF2_Tests)
    tests += list_test_cases(S2V_Tests)
    tests += list_test_cases(HKDF_Tests)
    tests += [TestVectorsHKDFWycheproof(wycheproof_warnings)]
    tests += list_test_cases(scrypt_Tests)
    tests += list_test_cases(bcrypt_Tests)

    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
