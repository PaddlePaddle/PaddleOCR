#
#  SelfTest/Signature/test_dss.py: Self-test for DSS signatures
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

import re
import unittest
from binascii import hexlify, unhexlify

from Crypto.Util.py3compat import tobytes, bord, bchr

from Crypto.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512,
                         SHA3_224, SHA3_256, SHA3_384, SHA3_512)
from Crypto.Signature import DSS
from Crypto.PublicKey import DSA, ECC
from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof
from Crypto.Util.number import bytes_to_long, long_to_bytes


def t2b(hexstring):
    ws = hexstring.replace(" ", "").replace("\n", "")
    return unhexlify(tobytes(ws))


def t2l(hexstring):
    ws = hexstring.replace(" ", "").replace("\n", "")
    return int(ws, 16)


def load_hash_by_name(hash_name):
    return __import__("Crypto.Hash." + hash_name, globals(), locals(), ["new"])


class StrRNG:

    def __init__(self, randomness):
        length = len(randomness)
        self._idx = 0
        # Fix required to get the right K (see how randint() works!)
        self._randomness = long_to_bytes(bytes_to_long(randomness) - 1, length)

    def __call__(self, n):
        out = self._randomness[self._idx:self._idx + n]
        self._idx += n
        return out


class FIPS_DSA_Tests(unittest.TestCase):

    # 1st 1024 bit key from SigGen.txt
    P = 0xa8f9cd201e5e35d892f85f80e4db2599a5676a3b1d4f190330ed3256b26d0e80a0e49a8fffaaad2a24f472d2573241d4d6d6c7480c80b4c67bb4479c15ada7ea8424d2502fa01472e760241713dab025ae1b02e1703a1435f62ddf4ee4c1b664066eb22f2e3bf28bb70a2a76e4fd5ebe2d1229681b5b06439ac9c7e9d8bde283
    Q = 0xf85f0f83ac4df7ea0cdf8f469bfeeaea14156495
    G = 0x2b3152ff6c62f14622b8f48e59f8af46883b38e79b8c74deeae9df131f8b856e3ad6c8455dab87cc0da8ac973417ce4f7878557d6cdf40b35b4a0ca3eb310c6a95d68ce284ad4e25ea28591611ee08b8444bd64b25f3f7c572410ddfb39cc728b9c936f85f419129869929cdb909a6a3a99bbe089216368171bd0ba81de4fe33
    X = 0xc53eae6d45323164c7d07af5715703744a63fc3a
    Y = 0x313fd9ebca91574e1c2eebe1517c57e0c21b0209872140c5328761bbb2450b33f1b18b409ce9ab7c4cd8fda3391e8e34868357c199e16a6b2eba06d6749def791d79e95d3a4d09b24c392ad89dbf100995ae19c01062056bb14bce005e8731efde175f95b975089bdcdaea562b32786d96f5a31aedf75364008ad4fffebb970b

    key_pub = DSA.construct((Y, G, P, Q))
    key_priv = DSA.construct((Y, G, P, Q, X))

    def shortDescription(self):
        return "FIPS DSA Tests"

    def test_loopback(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv, 'fips-186-3')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub, 'fips-186-3')
        verifier.verify(hashed_msg, signature)

    def test_negative_unapproved_hashes(self):
        """Verify that unapproved hashes are rejected"""

        from Crypto.Hash import RIPEMD160

        self.description = "Unapproved hash (RIPEMD160) test"
        hash_obj = RIPEMD160.new()
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertRaises(ValueError, signer.sign, hash_obj)
        self.assertRaises(ValueError, signer.verify, hash_obj, b"\x00" * 40)

    def test_negative_unknown_modes_encodings(self):
        """Verify that unknown modes/encodings are rejected"""

        self.description = "Unknown mode test"
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-0')

        self.description = "Unknown encoding test"
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-3', 'xml')

    def test_asn1_encoding(self):
        """Verify ASN.1 encoding"""

        self.description = "ASN.1 encoding test"
        hash_obj = SHA1.new()
        signer = DSS.new(self.key_priv, 'fips-186-3', 'der')
        signature = signer.sign(hash_obj)

        # Verify that output looks like a DER SEQUENCE
        self.assertEqual(bord(signature[0]), 48)
        signer.verify(hash_obj, signature)

        # Verify that ASN.1 parsing fails as expected
        signature = bchr(7) + signature[1:]
        self.assertRaises(ValueError, signer.verify, hash_obj, signature)

    def test_sign_verify(self):
        """Verify public/private method"""

        self.description = "can_sign() test"
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertTrue(signer.can_sign())

        signer = DSS.new(self.key_pub, 'fips-186-3')
        self.assertFalse(signer.can_sign())

        try:
            signer.sign(SHA256.new(b'xyz'))
        except TypeError as e:
            msg = str(e)
        else:
            msg = ""
        self.assertTrue("Private key is needed" in msg)


class FIPS_DSA_Tests_KAT(unittest.TestCase):
    pass


test_vectors_verify = load_test_vectors(("Signature", "DSA"),
                                        "FIPS_186_3_SigVer.rsp",
                                        "Signature Verification 186-3",
                                        {'result': lambda x: x}) or []

for idx, tv in enumerate(test_vectors_verify):

    if isinstance(tv, str):
        res = re.match(r"\[mod = L=([0-9]+), N=([0-9]+), ([a-zA-Z0-9-]+)\]", tv)
        assert(res)
        hash_name = res.group(3).replace("-", "")
        hash_module = load_hash_by_name(hash_name)
        continue

    if hasattr(tv, "p"):
        modulus = tv.p
        generator = tv.g
        suborder = tv.q
        continue

    hash_obj = hash_module.new(tv.msg)

    comps = [bytes_to_long(x) for x in (tv.y, generator, modulus, suborder)]
    key = DSA.construct(comps, False)  # type: ignore
    verifier = DSS.new(key, 'fips-186-3')

    def positive_test(self, verifier=verifier, hash_obj=hash_obj, signature=tv.r+tv.s):
        verifier.verify(hash_obj, signature)

    def negative_test(self, verifier=verifier, hash_obj=hash_obj, signature=tv.r+tv.s):
        self.assertRaises(ValueError, verifier.verify, hash_obj, signature)

    if tv.result == 'p':
        setattr(FIPS_DSA_Tests_KAT, "test_verify_positive_%d" % idx, positive_test)
    else:
        setattr(FIPS_DSA_Tests_KAT, "test_verify_negative_%d" % idx, negative_test)


test_vectors_sign = load_test_vectors(("Signature", "DSA"),
                                        "FIPS_186_3_SigGen.txt",
                                        "Signature Creation 186-3",
                                        {}) or []

for idx, tv in enumerate(test_vectors_sign):

    if isinstance(tv, str):
        res = re.match(r"\[mod = L=([0-9]+), N=([0-9]+), ([a-zA-Z0-9-]+)\]", tv)
        assert(res)
        hash_name = res.group(3).replace("-", "")
        hash_module = load_hash_by_name(hash_name)
        continue

    if hasattr(tv, "p"):
        modulus = tv.p
        generator = tv.g
        suborder = tv.q
        continue

    hash_obj = hash_module.new(tv.msg)
    comps_dsa = [bytes_to_long(x) for x in (tv.y, generator, modulus, suborder, tv.x)]
    key = DSA.construct(comps_dsa, False) # type: ignore
    signer = DSS.new(key, 'fips-186-3', randfunc=StrRNG(tv.k))

    def new_test(self, signer=signer, hash_obj=hash_obj, signature=tv.r+tv.s):
        self.assertEqual(signer.sign(hash_obj), signature)
    setattr(FIPS_DSA_Tests_KAT, "test_sign_%d" % idx, new_test)


class FIPS_ECDSA_Tests(unittest.TestCase):

    key_priv = ECC.generate(curve="P-256")
    key_pub = key_priv.public_key()

    def shortDescription(self):
        return "FIPS ECDSA Tests"

    def test_loopback(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv, 'fips-186-3')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub, 'fips-186-3')
        verifier.verify(hashed_msg, signature)

    def test_negative_unapproved_hashes(self):
        """Verify that unapproved hashes are rejected"""

        from Crypto.Hash import SHA1

        self.description = "Unapproved hash (SHA-1) test"
        hash_obj = SHA1.new()
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertRaises(ValueError, signer.sign, hash_obj)
        self.assertRaises(ValueError, signer.verify, hash_obj, b"\x00" * 40)

    def test_sign_verify(self):
        """Verify public/private method"""

        self.description = "can_sign() test"
        signer = DSS.new(self.key_priv, 'fips-186-3')
        self.assertTrue(signer.can_sign())

        signer = DSS.new(self.key_pub, 'fips-186-3')
        self.assertFalse(signer.can_sign())
        self.assertRaises(TypeError, signer.sign, SHA256.new(b'xyz'))

        try:
            signer.sign(SHA256.new(b'xyz'))
        except TypeError as e:
            msg = str(e)
        else:
            msg = ""
        self.assertTrue("Private key is needed" in msg)

    def test_negative_unknown_modes_encodings(self):
        """Verify that unknown modes/encodings are rejected"""

        self.description = "Unknown mode test"
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-0')

        self.description = "Unknown encoding test"
        self.assertRaises(ValueError, DSS.new, self.key_priv, 'fips-186-3', 'xml')

    def test_asn1_encoding(self):
        """Verify ASN.1 encoding"""

        self.description = "ASN.1 encoding test"
        hash_obj = SHA256.new()
        signer = DSS.new(self.key_priv, 'fips-186-3', 'der')
        signature = signer.sign(hash_obj)

        # Verify that output looks like a DER SEQUENCE
        self.assertEqual(bord(signature[0]), 48)
        signer.verify(hash_obj, signature)

        # Verify that ASN.1 parsing fails as expected
        signature = bchr(7) + signature[1:]
        self.assertRaises(ValueError, signer.verify, hash_obj, signature)


class FIPS_ECDSA_Tests_KAT(unittest.TestCase):
    pass


test_vectors_verify = load_test_vectors(("Signature", "ECDSA"),
                                        "SigVer.rsp",
                                        "ECDSA Signature Verification 186-3",
                                        {'result': lambda x: x,
                                         'qx': lambda x: int(x, 16),
                                         'qy': lambda x: int(x, 16),
                                        }) or []
test_vectors_verify += load_test_vectors(("Signature", "ECDSA"),
                                        "SigVer_TruncatedSHAs.rsp",
                                        "ECDSA Signature Verification 186-3",
                                        {'result': lambda x: x,
                                         'qx': lambda x: int(x, 16),
                                         'qy': lambda x: int(x, 16),
                                        }) or []


for idx, tv in enumerate(test_vectors_verify):

    if isinstance(tv, str):
        res = re.match(r"\[(P-[0-9]+),(SHA-[0-9]+)\]", tv)
        assert res
        curve_name = res.group(1)
        hash_name = res.group(2).replace("-", "")
        if hash_name in ("SHA512224", "SHA512256"):
            truncate = hash_name[-3:]
            hash_name = hash_name[:-3]
        else:
            truncate = None
        hash_module = load_hash_by_name(hash_name)
        continue

    if truncate is None:
        hash_obj = hash_module.new(tv.msg)
    else:
        hash_obj = hash_module.new(tv.msg, truncate=truncate)
    ecc_key = ECC.construct(curve=curve_name, point_x=tv.qx, point_y=tv.qy)
    verifier = DSS.new(ecc_key, 'fips-186-3')

    def positive_test(self, verifier=verifier, hash_obj=hash_obj, signature=tv.r+tv.s):
        verifier.verify(hash_obj, signature)

    def negative_test(self, verifier=verifier, hash_obj=hash_obj, signature=tv.r+tv.s):
        self.assertRaises(ValueError, verifier.verify, hash_obj, signature)

    if tv.result.startswith('p'):
        setattr(FIPS_ECDSA_Tests_KAT, "test_verify_positive_%d" % idx, positive_test)
    else:
        setattr(FIPS_ECDSA_Tests_KAT, "test_verify_negative_%d" % idx, negative_test)


test_vectors_sign = load_test_vectors(("Signature", "ECDSA"),
                                        "SigGen.txt",
                                        "ECDSA Signature Verification 186-3",
                                        {'d': lambda x: int(x, 16)}) or []

for idx, tv in enumerate(test_vectors_sign):

    if isinstance(tv, str):
        res = re.match(r"\[(P-[0-9]+),(SHA-[0-9]+)\]", tv)
        assert res
        curve_name = res.group(1)
        hash_name = res.group(2).replace("-", "")
        hash_module = load_hash_by_name(hash_name)
        continue

    hash_obj = hash_module.new(tv.msg)
    ecc_key = ECC.construct(curve=curve_name, d=tv.d)
    signer = DSS.new(ecc_key, 'fips-186-3', randfunc=StrRNG(tv.k))

    def sign_test(self, signer=signer, hash_obj=hash_obj, signature=tv.r+tv.s):
        self.assertEqual(signer.sign(hash_obj), signature)
    setattr(FIPS_ECDSA_Tests_KAT, "test_sign_%d" % idx, sign_test)


class Det_DSA_Tests(unittest.TestCase):
    """Tests from rfc6979"""

    # Each key is (p, q, g, x, y, desc)
    keys = [
            (
            """
            86F5CA03DCFEB225063FF830A0C769B9DD9D6153AD91D7CE27F787C43278B447
            E6533B86B18BED6E8A48B784A14C252C5BE0DBF60B86D6385BD2F12FB763ED88
            73ABFD3F5BA2E0A8C0A59082EAC056935E529DAF7C610467899C77ADEDFC846C
            881870B7B19B2B58F9BE0521A17002E3BDD6B86685EE90B3D9A1B02B782B1779""",
            "996F967F6C8E388D9E28D01E205FBA957A5698B1",
            """
            07B0F92546150B62514BB771E2A0C0CE387F03BDA6C56B505209FF25FD3C133D
            89BBCD97E904E09114D9A7DEFDEADFC9078EA544D2E401AEECC40BB9FBBF78FD
            87995A10A1C27CB7789B594BA7EFB5C4326A9FE59A070E136DB77175464ADCA4
            17BE5DCE2F40D10A46A3A3943F26AB7FD9C0398FF8C76EE0A56826A8A88F1DBD""",
            "411602CB19A6CCC34494D79D98EF1E7ED5AF25F7",
            """
            5DF5E01DED31D0297E274E1691C192FE5868FEF9E19A84776454B100CF16F653
            92195A38B90523E2542EE61871C0440CB87C322FC4B4D2EC5E1E7EC766E1BE8D
            4CE935437DC11C3C8FD426338933EBFE739CB3465F4D3668C5E473508253B1E6
            82F65CBDC4FAE93C2EA212390E54905A86E2223170B44EAA7DA5DD9FFCFB7F3B""",
            "DSA1024"
            ),
            (
            """
            9DB6FB5951B66BB6FE1E140F1D2CE5502374161FD6538DF1648218642F0B5C48
            C8F7A41AADFA187324B87674FA1822B00F1ECF8136943D7C55757264E5A1A44F
            FE012E9936E00C1D3E9310B01C7D179805D3058B2A9F4BB6F9716BFE6117C6B5
            B3CC4D9BE341104AD4A80AD6C94E005F4B993E14F091EB51743BF33050C38DE2
            35567E1B34C3D6A5C0CEAA1A0F368213C3D19843D0B4B09DCB9FC72D39C8DE41
            F1BF14D4BB4563CA28371621CAD3324B6A2D392145BEBFAC748805236F5CA2FE
            92B871CD8F9C36D3292B5509CA8CAA77A2ADFC7BFD77DDA6F71125A7456FEA15
            3E433256A2261C6A06ED3693797E7995FAD5AABBCFBE3EDA2741E375404AE25B""",
            "F2C3119374CE76C9356990B465374A17F23F9ED35089BD969F61C6DDE9998C1F",
            """
            5C7FF6B06F8F143FE8288433493E4769C4D988ACE5BE25A0E24809670716C613
            D7B0CEE6932F8FAA7C44D2CB24523DA53FBE4F6EC3595892D1AA58C4328A06C4
            6A15662E7EAA703A1DECF8BBB2D05DBE2EB956C142A338661D10461C0D135472
            085057F3494309FFA73C611F78B32ADBB5740C361C9F35BE90997DB2014E2EF5
            AA61782F52ABEB8BD6432C4DD097BC5423B285DAFB60DC364E8161F4A2A35ACA
            3A10B1C4D203CC76A470A33AFDCBDD92959859ABD8B56E1725252D78EAC66E71
            BA9AE3F1DD2487199874393CD4D832186800654760E1E34C09E4D155179F9EC0
            DC4473F996BDCE6EED1CABED8B6F116F7AD9CF505DF0F998E34AB27514B0FFE7""",
            "69C7548C21D0DFEA6B9A51C9EAD4E27C33D3B3F180316E5BCAB92C933F0E4DBC",
            """
            667098C654426C78D7F8201EAC6C203EF030D43605032C2F1FA937E5237DBD94
            9F34A0A2564FE126DC8B715C5141802CE0979C8246463C40E6B6BDAA2513FA61
            1728716C2E4FD53BC95B89E69949D96512E873B9C8F8DFD499CC312882561ADE
            CB31F658E934C0C197F2C4D96B05CBAD67381E7B768891E4DA3843D24D94CDFB
            5126E9B8BF21E8358EE0E0A30EF13FD6A664C0DCE3731F7FB49A4845A4FD8254
            687972A2D382599C9BAC4E0ED7998193078913032558134976410B89D2C171D1
            23AC35FD977219597AA7D15C1A9A428E59194F75C721EBCBCFAE44696A499AFA
            74E04299F132026601638CB87AB79190D4A0986315DA8EEC6561C938996BEADF""",
            "DSA2048"
            ),
           ]

    # This is a sequence of items:
    # message, k, r, s, hash module
    signatures = [
            (
                "sample",
                "7BDB6B0FF756E1BB5D53583EF979082F9AD5BD5B",
                "2E1A0C2562B2912CAAF89186FB0F42001585DA55",
                "29EFB6B0AFF2D7A68EB70CA313022253B9A88DF5",
                SHA1,
                'DSA1024'
            ),
            (
                "sample",
                "562097C06782D60C3037BA7BE104774344687649",
                "4BC3B686AEA70145856814A6F1BB53346F02101E",
                "410697B92295D994D21EDD2F4ADA85566F6F94C1",
                SHA224,
                'DSA1024'
            ),
            (
                "sample",
                "519BA0546D0C39202A7D34D7DFA5E760B318BCFB",
                "81F2F5850BE5BC123C43F71A3033E9384611C545",
                "4CDD914B65EB6C66A8AAAD27299BEE6B035F5E89",
                SHA256,
                'DSA1024'
            ),
            (
                "sample",
                "95897CD7BBB944AA932DBC579C1C09EB6FCFC595",
                "07F2108557EE0E3921BC1774F1CA9B410B4CE65A",
                "54DF70456C86FAC10FAB47C1949AB83F2C6F7595",
                SHA384,
                'DSA1024'
            ),
            (
                "sample",
                "09ECE7CA27D0F5A4DD4E556C9DF1D21D28104F8B",
                "16C3491F9B8C3FBBDD5E7A7B667057F0D8EE8E1B",
                "02C36A127A7B89EDBB72E4FFBC71DABC7D4FC69C",
                SHA512,
                'DSA1024'
            ),
            (
                "test",
                "5C842DF4F9E344EE09F056838B42C7A17F4A6433",
                "42AB2052FD43E123F0607F115052A67DCD9C5C77",
                "183916B0230D45B9931491D4C6B0BD2FB4AAF088",
                SHA1,
                'DSA1024'
            ),
            (
                "test",
                "4598B8EFC1A53BC8AECD58D1ABBB0C0C71E67297",
                "6868E9964E36C1689F6037F91F28D5F2C30610F2",
                "49CEC3ACDC83018C5BD2674ECAAD35B8CD22940F",
                SHA224,
                'DSA1024'
            ),
            (
                "test",
                "5A67592E8128E03A417B0484410FB72C0B630E1A",
                "22518C127299B0F6FDC9872B282B9E70D0790812",
                "6837EC18F150D55DE95B5E29BE7AF5D01E4FE160",
                SHA256,
                'DSA1024'
            ),
            (
                "test",
                "220156B761F6CA5E6C9F1B9CF9C24BE25F98CD89",
                "854CF929B58D73C3CBFDC421E8D5430CD6DB5E66",
                "91D0E0F53E22F898D158380676A871A157CDA622",
                SHA384,
                'DSA1024'
            ),
            (
                "test",
                "65D2C2EEB175E370F28C75BFCDC028D22C7DBE9C",
                "8EA47E475BA8AC6F2D821DA3BD212D11A3DEB9A0",
                "7C670C7AD72B6C050C109E1790008097125433E8",
                SHA512,
                'DSA1024'
            ),
            (
                "sample",
                "888FA6F7738A41BDC9846466ABDB8174C0338250AE50CE955CA16230F9CBD53E",
                "3A1B2DBD7489D6ED7E608FD036C83AF396E290DBD602408E8677DAABD6E7445A",
                "D26FCBA19FA3E3058FFC02CA1596CDBB6E0D20CB37B06054F7E36DED0CDBBCCF",
                SHA1,
                'DSA2048'
            ),
            (
                "sample",
                "BC372967702082E1AA4FCE892209F71AE4AD25A6DFD869334E6F153BD0C4D806",
                "DC9F4DEADA8D8FF588E98FED0AB690FFCE858DC8C79376450EB6B76C24537E2C",
                "A65A9C3BC7BABE286B195D5DA68616DA8D47FA0097F36DD19F517327DC848CEC",
                SHA224,
                'DSA2048'
            ),
            (
                "sample",
                "8926A27C40484216F052F4427CFD5647338B7B3939BC6573AF4333569D597C52",
                "EACE8BDBBE353C432A795D9EC556C6D021F7A03F42C36E9BC87E4AC7932CC809",
                "7081E175455F9247B812B74583E9E94F9EA79BD640DC962533B0680793A38D53",
                SHA256,
                'DSA2048'
            ),
            (
                "sample",
                "C345D5AB3DA0A5BCB7EC8F8FB7A7E96069E03B206371EF7D83E39068EC564920",
                "B2DA945E91858834FD9BF616EBAC151EDBC4B45D27D0DD4A7F6A22739F45C00B",
                "19048B63D9FD6BCA1D9BAE3664E1BCB97F7276C306130969F63F38FA8319021B",
                SHA384,
                'DSA2048'
            ),
            (
                "sample",
                "5A12994431785485B3F5F067221517791B85A597B7A9436995C89ED0374668FC",
                "2016ED092DC5FB669B8EFB3D1F31A91EECB199879BE0CF78F02BA062CB4C942E",
                "D0C76F84B5F091E141572A639A4FB8C230807EEA7D55C8A154A224400AFF2351",
                SHA512,
                'DSA2048'
            ),
            (
                "test",
                "6EEA486F9D41A037B2C640BC5645694FF8FF4B98D066A25F76BE641CCB24BA4F",
                "C18270A93CFC6063F57A4DFA86024F700D980E4CF4E2CB65A504397273D98EA0",
                "414F22E5F31A8B6D33295C7539C1C1BA3A6160D7D68D50AC0D3A5BEAC2884FAA",
                SHA1,
                'DSA2048'
            ),
            (
                "test",
                "06BD4C05ED74719106223BE33F2D95DA6B3B541DAD7BFBD7AC508213B6DA6670",
                "272ABA31572F6CC55E30BF616B7A265312018DD325BE031BE0CC82AA17870EA3",
                "E9CC286A52CCE201586722D36D1E917EB96A4EBDB47932F9576AC645B3A60806",
                SHA224,
                'DSA2048'
            ),
            (
                "test",
                "1D6CE6DDA1C5D37307839CD03AB0A5CBB18E60D800937D67DFB4479AAC8DEAD7",
                "8190012A1969F9957D56FCCAAD223186F423398D58EF5B3CEFD5A4146A4476F0",
                "7452A53F7075D417B4B013B278D1BB8BBD21863F5E7B1CEE679CF2188E1AB19E",
                SHA256,
                'DSA2048'
            ),
            (
                "test",
                "206E61F73DBE1B2DC8BE736B22B079E9DACD974DB00EEBBC5B64CAD39CF9F91C",
                "239E66DDBE8F8C230A3D071D601B6FFBDFB5901F94D444C6AF56F732BEB954BE",
                "6BD737513D5E72FE85D1C750E0F73921FE299B945AAD1C802F15C26A43D34961",
                SHA384,
                'DSA2048'
            ),
            (
                "test",
                "AFF1651E4CD6036D57AA8B2A05CCF1A9D5A40166340ECBBDC55BE10B568AA0AA",
                "89EC4BB1400ECCFF8E7D9AA515CD1DE7803F2DAFF09693EE7FD1353E90A68307",
                "C9F0BDABCC0D880BB137A994CC7F3980CE91CC10FAF529FC46565B15CEA854E1",
                SHA512,
                'DSA2048'
            )
        ]

    def setUp(self):
        # Convert DSA key components from hex strings to integers
        # Each key is (p, q, g, x, y, desc)

        from collections import namedtuple

        TestKey = namedtuple('TestKey', 'p q g x y')
        new_keys = {}
        for k in self.keys:
            tk = TestKey(*[t2l(y) for y in k[:-1]])
            new_keys[k[-1]] = tk
        self.keys = new_keys

        # Convert signature encoding
        TestSig = namedtuple('TestSig', 'message nonce result module test_key')
        new_signatures = []
        for message, nonce, r, s, module, test_key in self.signatures:
            tsig = TestSig(
                tobytes(message),
                t2l(nonce),
                t2b(r) + t2b(s),
                module,
                self.keys[test_key]
            )
            new_signatures.append(tsig)
        self.signatures = new_signatures

    def test1(self):
        q = 0x4000000000000000000020108A2E0CC0D99F8A5EF
        x = 0x09A4D6792295A7F730FC3F2B49CBC0F62E862272F
        p = 2 * q + 1
        y = pow(2, x, p)
        key = DSA.construct([pow(y, 2, p), 2, p, q, x], False)
        signer = DSS.new(key, 'deterministic-rfc6979')

        # Test _int2octets
        self.assertEqual(hexlify(signer._int2octets(x)),
            b'009a4d6792295a7f730fc3f2b49cbc0f62e862272f')

        # Test _bits2octets
        h1 = SHA256.new(b"sample").digest()
        self.assertEqual(hexlify(signer._bits2octets(h1)),
            b'01795edf0d54db760f156d0dac04c0322b3a204224')

    def test2(self):

        for sig in self.signatures:
            tk = sig.test_key
            key = DSA.construct([tk.y, tk.g, tk.p, tk.q, tk.x], False)
            signer = DSS.new(key, 'deterministic-rfc6979')

            hash_obj = sig.module.new(sig.message)
            result = signer.sign(hash_obj)
            self.assertEqual(sig.result, result)


class Det_ECDSA_Tests(unittest.TestCase):

    key_priv_p192 = ECC.construct(curve="P-192", d=0x6FAB034934E4C0FC9AE67F5B5659A9D7D1FEFD187EE09FD4)
    key_pub_p192 = key_priv_p192.public_key()

    key_priv_p224 = ECC.construct(curve="P-224", d=0xF220266E1105BFE3083E03EC7A3A654651F45E37167E88600BF257C1)
    key_pub_p224 = key_priv_p224.public_key()

    key_priv_p256 = ECC.construct(curve="P-256", d=0xC9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721)
    key_pub_p256 = key_priv_p256.public_key()

    key_priv_p384 = ECC.construct(curve="P-384", d=0x6B9D3DAD2E1B8C1C05B19875B6659F4DE23C3B667BF297BA9AA47740787137D896D5724E4C70A825F872C9EA60D2EDF5)
    key_pub_p384 = key_priv_p384.public_key()

    key_priv_p521 = ECC.construct(curve="P-521", d=0x0FAD06DAA62BA3B25D2FB40133DA757205DE67F5BB0018FEE8C86E1B68C7E75CAA896EB32F1F47C70855836A6D16FCC1466F6D8FBEC67DB89EC0C08B0E996B83538)
    key_pub_p521 = key_priv_p521.public_key()

    # This is a sequence of items:
    # message, k, r, s, hash module
    # taken from RFC6979
    signatures_p192_ = (
        (
            "sample",
            "37D7CA00D2C7B0E5E412AC03BD44BA837FDD5B28CD3B0021",
            "98C6BD12B23EAF5E2A2045132086BE3EB8EBD62ABF6698FF",
            "57A22B07DEA9530F8DE9471B1DC6624472E8E2844BC25B64",
            SHA1
        ),
        (
            "sample",
            "4381526B3FC1E7128F202E194505592F01D5FF4C5AF015D8",
            "A1F00DAD97AEEC91C95585F36200C65F3C01812AA60378F5",
            "E07EC1304C7C6C9DEBBE980B9692668F81D4DE7922A0F97A",
            SHA224
        ),
        (
            "sample",
            "32B1B6D7D42A05CB449065727A84804FB1A3E34D8F261496",
            "4B0B8CE98A92866A2820E20AA6B75B56382E0F9BFD5ECB55",
            "CCDB006926EA9565CBADC840829D8C384E06DE1F1E381B85",
            SHA256
        ),
        (
            "sample",
            "4730005C4FCB01834C063A7B6760096DBE284B8252EF4311",
            "DA63BF0B9ABCF948FBB1E9167F136145F7A20426DCC287D5",
            "C3AA2C960972BD7A2003A57E1C4C77F0578F8AE95E31EC5E",
            SHA384
        ),
        (
            "sample",
            "A2AC7AB055E4F20692D49209544C203A7D1F2C0BFBC75DB1",
            "4D60C5AB1996BD848343B31C00850205E2EA6922DAC2E4B8",
            "3F6E837448F027A1BF4B34E796E32A811CBB4050908D8F67",
            SHA512
        ),
        (
            "test",
            "D9CF9C3D3297D3260773A1DA7418DB5537AB8DD93DE7FA25",
            "0F2141A0EBBC44D2E1AF90A50EBCFCE5E197B3B7D4DE036D",
            "EB18BC9E1F3D7387500CB99CF5F7C157070A8961E38700B7",
            SHA1
        ),
        (
            "test",
            "F5DC805F76EF851800700CCE82E7B98D8911B7D510059FBE",
            "6945A1C1D1B2206B8145548F633BB61CEF04891BAF26ED34",
            "B7FB7FDFC339C0B9BD61A9F5A8EAF9BE58FC5CBA2CB15293",
            SHA224
        ),
        (
            "test",
            "5C4CE89CF56D9E7C77C8585339B006B97B5F0680B4306C6C",
            "3A718BD8B4926C3B52EE6BBE67EF79B18CB6EB62B1AD97AE",
            "5662E6848A4A19B1F1AE2F72ACD4B8BBE50F1EAC65D9124F",
            SHA256
        ),
        (
            "test",
            "5AFEFB5D3393261B828DB6C91FBC68C230727B030C975693",
            "B234B60B4DB75A733E19280A7A6034BD6B1EE88AF5332367",
            "7994090B2D59BB782BE57E74A44C9A1C700413F8ABEFE77A",
            SHA384
        ),
        (
            "test",
            "0758753A5254759C7CFBAD2E2D9B0792EEE44136C9480527",
            "FE4F4AE86A58B6507946715934FE2D8FF9D95B6B098FE739",
            "74CF5605C98FBA0E1EF34D4B5A1577A7DCF59457CAE52290",
            SHA512
        )
    )

    signatures_p224_ = (
        (
            "sample",
            "7EEFADD91110D8DE6C2C470831387C50D3357F7F4D477054B8B426BC",
            "22226F9D40A96E19C4A301CE5B74B115303C0F3A4FD30FC257FB57AC",
            "66D1CDD83E3AF75605DD6E2FEFF196D30AA7ED7A2EDF7AF475403D69",
            SHA1
        ),
        (
            "sample",
            "C1D1F2F10881088301880506805FEB4825FE09ACB6816C36991AA06D",
            "1CDFE6662DDE1E4A1EC4CDEDF6A1F5A2FB7FBD9145C12113E6ABFD3E",
            "A6694FD7718A21053F225D3F46197CA699D45006C06F871808F43EBC",
            SHA224
        ),
        (
            "sample",
            "AD3029E0278F80643DE33917CE6908C70A8FF50A411F06E41DEDFCDC",
            "61AA3DA010E8E8406C656BC477A7A7189895E7E840CDFE8FF42307BA",
            "BC814050DAB5D23770879494F9E0A680DC1AF7161991BDE692B10101",
            SHA256
        ),
        (
            "sample",
            "52B40F5A9D3D13040F494E83D3906C6079F29981035C7BD51E5CAC40",
            "0B115E5E36F0F9EC81F1325A5952878D745E19D7BB3EABFABA77E953",
            "830F34CCDFE826CCFDC81EB4129772E20E122348A2BBD889A1B1AF1D",
            SHA384
        ),
        (
            "sample",
            "9DB103FFEDEDF9CFDBA05184F925400C1653B8501BAB89CEA0FBEC14",
            "074BD1D979D5F32BF958DDC61E4FB4872ADCAFEB2256497CDAC30397",
            "A4CECA196C3D5A1FF31027B33185DC8EE43F288B21AB342E5D8EB084",
            SHA512
        ),
        (
            "test",
            "2519178F82C3F0E4F87ED5883A4E114E5B7A6E374043D8EFD329C253",
            "DEAA646EC2AF2EA8AD53ED66B2E2DDAA49A12EFD8356561451F3E21C",
            "95987796F6CF2062AB8135271DE56AE55366C045F6D9593F53787BD2",
            SHA1
        ),
        (
            "test",
            "DF8B38D40DCA3E077D0AC520BF56B6D565134D9B5F2EAE0D34900524",
            "C441CE8E261DED634E4CF84910E4C5D1D22C5CF3B732BB204DBEF019",
            "902F42847A63BDC5F6046ADA114953120F99442D76510150F372A3F4",
            SHA224
        ),
        (
            "test",
            "FF86F57924DA248D6E44E8154EB69F0AE2AEBAEE9931D0B5A969F904",
            "AD04DDE87B84747A243A631EA47A1BA6D1FAA059149AD2440DE6FBA6",
            "178D49B1AE90E3D8B629BE3DB5683915F4E8C99FDF6E666CF37ADCFD",
            SHA256
        ),
        (
            "test",
            "7046742B839478C1B5BD31DB2E862AD868E1A45C863585B5F22BDC2D",
            "389B92682E399B26518A95506B52C03BC9379A9DADF3391A21FB0EA4",
            "414A718ED3249FF6DBC5B50C27F71F01F070944DA22AB1F78F559AAB",
            SHA384
        ),
        (
            "test",
            "E39C2AA4EA6BE2306C72126D40ED77BF9739BB4D6EF2BBB1DCB6169D",
            "049F050477C5ADD858CAC56208394B5A55BAEBBE887FDF765047C17C",
            "077EB13E7005929CEFA3CD0403C7CDCC077ADF4E44F3C41B2F60ECFF",
            SHA512
        )
    )

    signatures_p256_ = (
        (
            "sample",
            "882905F1227FD620FBF2ABF21244F0BA83D0DC3A9103DBBEE43A1FB858109DB4",
            "61340C88C3AAEBEB4F6D667F672CA9759A6CCAA9FA8811313039EE4A35471D32",
            "6D7F147DAC089441BB2E2FE8F7A3FA264B9C475098FDCF6E00D7C996E1B8B7EB",
            SHA1
        ),
        (
            "sample",
            "103F90EE9DC52E5E7FB5132B7033C63066D194321491862059967C715985D473",
            "53B2FFF5D1752B2C689DF257C04C40A587FABABB3F6FC2702F1343AF7CA9AA3F",
            "B9AFB64FDC03DC1A131C7D2386D11E349F070AA432A4ACC918BEA988BF75C74C",
            SHA224
        ),
        (
            "sample",
            "A6E3C57DD01ABE90086538398355DD4C3B17AA873382B0F24D6129493D8AAD60",
            "EFD48B2AACB6A8FD1140DD9CD45E81D69D2C877B56AAF991C34D0EA84EAF3716",
            "F7CB1C942D657C41D436C7A1B6E29F65F3E900DBB9AFF4064DC4AB2F843ACDA8",
            SHA256
        ),
        (
            "sample",
            "09F634B188CEFD98E7EC88B1AA9852D734D0BC272F7D2A47DECC6EBEB375AAD4",
            "0EAFEA039B20E9B42309FB1D89E213057CBF973DC0CFC8F129EDDDC800EF7719",
            "4861F0491E6998B9455193E34E7B0D284DDD7149A74B95B9261F13ABDE940954",
            SHA384
        ),
        (
            "sample",
            "5FA81C63109BADB88C1F367B47DA606DA28CAD69AA22C4FE6AD7DF73A7173AA5",
            "8496A60B5E9B47C825488827E0495B0E3FA109EC4568FD3F8D1097678EB97F00",
            "2362AB1ADBE2B8ADF9CB9EDAB740EA6049C028114F2460F96554F61FAE3302FE",
            SHA512
        ),
        (
            "test",
            "8C9520267C55D6B980DF741E56B4ADEE114D84FBFA2E62137954164028632A2E",
            "0CBCC86FD6ABD1D99E703E1EC50069EE5C0B4BA4B9AC60E409E8EC5910D81A89",
            "01B9D7B73DFAA60D5651EC4591A0136F87653E0FD780C3B1BC872FFDEAE479B1",
            SHA1
        ),
        (
            "test",
            "669F4426F2688B8BE0DB3A6BD1989BDAEFFF84B649EEB84F3DD26080F667FAA7",
            "C37EDB6F0AE79D47C3C27E962FA269BB4F441770357E114EE511F662EC34A692",
            "C820053A05791E521FCAAD6042D40AEA1D6B1A540138558F47D0719800E18F2D",
            SHA224
        ),
        (
            "test",
            "D16B6AE827F17175E040871A1C7EC3500192C4C92677336EC2537ACAEE0008E0",
            "F1ABB023518351CD71D881567B1EA663ED3EFCF6C5132B354F28D3B0B7D38367",
            "019F4113742A2B14BD25926B49C649155F267E60D3814B4C0CC84250E46F0083",
            SHA256
        ),
        (
            "test",
            "16AEFFA357260B04B1DD199693960740066C1A8F3E8EDD79070AA914D361B3B8",
            "83910E8B48BB0C74244EBDF7F07A1C5413D61472BD941EF3920E623FBCCEBEB6",
            "8DDBEC54CF8CD5874883841D712142A56A8D0F218F5003CB0296B6B509619F2C",
            SHA384
        ),
        (
            "test",
            "6915D11632ACA3C40D5D51C08DAF9C555933819548784480E93499000D9F0B7F",
            "461D93F31B6540894788FD206C07CFA0CC35F46FA3C91816FFF1040AD1581A04",
            "39AF9F15DE0DB8D97E72719C74820D304CE5226E32DEDAE67519E840D1194E55",
            SHA512
        )
    )

    signatures_p384_ = (
        (
            "sample",
            "4471EF7518BB2C7C20F62EAE1C387AD0C5E8E470995DB4ACF694466E6AB096630F29E5938D25106C3C340045A2DB01A7",
            "EC748D839243D6FBEF4FC5C4859A7DFFD7F3ABDDF72014540C16D73309834FA37B9BA002899F6FDA3A4A9386790D4EB2",
            "A3BCFA947BEEF4732BF247AC17F71676CB31A847B9FF0CBC9C9ED4C1A5B3FACF26F49CA031D4857570CCB5CA4424A443",
            SHA1
        ),
        (
            "sample",
            "A4E4D2F0E729EB786B31FC20AD5D849E304450E0AE8E3E341134A5C1AFA03CAB8083EE4E3C45B06A5899EA56C51B5879",
            "42356E76B55A6D9B4631C865445DBE54E056D3B3431766D0509244793C3F9366450F76EE3DE43F5A125333A6BE060122",
            "9DA0C81787064021E78DF658F2FBB0B042BF304665DB721F077A4298B095E4834C082C03D83028EFBF93A3C23940CA8D",
            SHA224
        ),
        (
            "sample",
            "180AE9F9AEC5438A44BC159A1FCB277C7BE54FA20E7CF404B490650A8ACC414E375572342863C899F9F2EDF9747A9B60",
            "21B13D1E013C7FA1392D03C5F99AF8B30C570C6F98D4EA8E354B63A21D3DAA33BDE1E888E63355D92FA2B3C36D8FB2CD",
            "F3AA443FB107745BF4BD77CB3891674632068A10CA67E3D45DB2266FA7D1FEEBEFDC63ECCD1AC42EC0CB8668A4FA0AB0",
            SHA256
        ),
        (
            "sample",
            "94ED910D1A099DAD3254E9242AE85ABDE4BA15168EAF0CA87A555FD56D10FBCA2907E3E83BA95368623B8C4686915CF9",
            "94EDBB92A5ECB8AAD4736E56C691916B3F88140666CE9FA73D64C4EA95AD133C81A648152E44ACF96E36DD1E80FABE46",
            "99EF4AEB15F178CEA1FE40DB2603138F130E740A19624526203B6351D0A3A94FA329C145786E679E7B82C71A38628AC8",
            SHA384
        ),
        (
            "sample",
            "92FC3C7183A883E24216D1141F1A8976C5B0DD797DFA597E3D7B32198BD35331A4E966532593A52980D0E3AAA5E10EC3",
            "ED0959D5880AB2D869AE7F6C2915C6D60F96507F9CB3E047C0046861DA4A799CFE30F35CC900056D7C99CD7882433709",
            "512C8CCEEE3890A84058CE1E22DBC2198F42323CE8ACA9135329F03C068E5112DC7CC3EF3446DEFCEB01A45C2667FDD5",
            SHA512
        ),
        (
            "test",
            "66CC2C8F4D303FC962E5FF6A27BD79F84EC812DDAE58CF5243B64A4AD8094D47EC3727F3A3C186C15054492E30698497",
            "4BC35D3A50EF4E30576F58CD96CE6BF638025EE624004A1F7789A8B8E43D0678ACD9D29876DAF46638645F7F404B11C7",
            "D5A6326C494ED3FF614703878961C0FDE7B2C278F9A65FD8C4B7186201A2991695BA1C84541327E966FA7B50F7382282",
            SHA1
        ),
        (
            "test",
            "18FA39DB95AA5F561F30FA3591DC59C0FA3653A80DAFFA0B48D1A4C6DFCBFF6E3D33BE4DC5EB8886A8ECD093F2935726",
            "E8C9D0B6EA72A0E7837FEA1D14A1A9557F29FAA45D3E7EE888FC5BF954B5E62464A9A817C47FF78B8C11066B24080E72",
            "07041D4A7A0379AC7232FF72E6F77B6DDB8F09B16CCE0EC3286B2BD43FA8C6141C53EA5ABEF0D8231077A04540A96B66",
            SHA224
        ),
        (
            "test",
            "0CFAC37587532347DC3389FDC98286BBA8C73807285B184C83E62E26C401C0FAA48DD070BA79921A3457ABFF2D630AD7",
            "6D6DEFAC9AB64DABAFE36C6BF510352A4CC27001263638E5B16D9BB51D451559F918EEDAF2293BE5B475CC8F0188636B",
            "2D46F3BECBCC523D5F1A1256BF0C9B024D879BA9E838144C8BA6BAEB4B53B47D51AB373F9845C0514EEFB14024787265",
            SHA256
        ),
        (
            "test",
            "015EE46A5BF88773ED9123A5AB0807962D193719503C527B031B4C2D225092ADA71F4A459BC0DA98ADB95837DB8312EA",
            "8203B63D3C853E8D77227FB377BCF7B7B772E97892A80F36AB775D509D7A5FEB0542A7F0812998DA8F1DD3CA3CF023DB",
            "DDD0760448D42D8A43AF45AF836FCE4DE8BE06B485E9B61B827C2F13173923E06A739F040649A667BF3B828246BAA5A5",
            SHA384
        ),
        (
            "test",
            "3780C4F67CB15518B6ACAE34C9F83568D2E12E47DEAB6C50A4E4EE5319D1E8CE0E2CC8A136036DC4B9C00E6888F66B6C",
            "A0D5D090C9980FAF3C2CE57B7AE951D31977DD11C775D314AF55F76C676447D06FB6495CD21B4B6E340FC236584FB277",
            "976984E59B4C77B0E8E4460DCA3D9F20E07B9BB1F63BEEFAF576F6B2E8B224634A2092CD3792E0159AD9CEE37659C736",
            SHA512
        ),
    )

    signatures_p521_ = (
        (
            "sample",
            "0089C071B419E1C2820962321787258469511958E80582E95D8378E0C2CCDB3CB42BEDE42F50E3FA3C71F5A76724281D31D9C89F0F91FC1BE4918DB1C03A5838D0F9",
            "00343B6EC45728975EA5CBA6659BBB6062A5FF89EEA58BE3C80B619F322C87910FE092F7D45BB0F8EEE01ED3F20BABEC079D202AE677B243AB40B5431D497C55D75D",
            "00E7B0E675A9B24413D448B8CC119D2BF7B2D2DF032741C096634D6D65D0DBE3D5694625FB9E8104D3B842C1B0E2D0B98BEA19341E8676AEF66AE4EBA3D5475D5D16",
            SHA1
        ),
        (
            "sample",
            "0121415EC2CD7726330A61F7F3FA5DE14BE9436019C4DB8CB4041F3B54CF31BE0493EE3F427FB906393D895A19C9523F3A1D54BB8702BD4AA9C99DAB2597B92113F3",
            "01776331CFCDF927D666E032E00CF776187BC9FDD8E69D0DABB4109FFE1B5E2A30715F4CC923A4A5E94D2503E9ACFED92857B7F31D7152E0F8C00C15FF3D87E2ED2E",
            "0050CB5265417FE2320BBB5A122B8E1A32BD699089851128E360E620A30C7E17BA41A666AF126CE100E5799B153B60528D5300D08489CA9178FB610A2006C254B41F",
            SHA224
        ),
        (
            "sample",
            "00EDF38AFCAAECAB4383358B34D67C9F2216C8382AAEA44A3DAD5FDC9C32575761793FEF24EB0FC276DFC4F6E3EC476752F043CF01415387470BCBD8678ED2C7E1A0",
            "01511BB4D675114FE266FC4372B87682BAECC01D3CC62CF2303C92B3526012659D16876E25C7C1E57648F23B73564D67F61C6F14D527D54972810421E7D87589E1A7",
            "004A171143A83163D6DF460AAF61522695F207A58B95C0644D87E52AA1A347916E4F7A72930B1BC06DBE22CE3F58264AFD23704CBB63B29B931F7DE6C9D949A7ECFC",
            SHA256
        ),
        (
            "sample",
            "01546A108BC23A15D6F21872F7DED661FA8431DDBD922D0DCDB77CC878C8553FFAD064C95A920A750AC9137E527390D2D92F153E66196966EA554D9ADFCB109C4211",
            "01EA842A0E17D2DE4F92C15315C63DDF72685C18195C2BB95E572B9C5136CA4B4B576AD712A52BE9730627D16054BA40CC0B8D3FF035B12AE75168397F5D50C67451",
            "01F21A3CEE066E1961025FB048BD5FE2B7924D0CD797BABE0A83B66F1E35EEAF5FDE143FA85DC394A7DEE766523393784484BDF3E00114A1C857CDE1AA203DB65D61",
            SHA384
        ),
        (
            "sample",
            "01DAE2EA071F8110DC26882D4D5EAE0621A3256FC8847FB9022E2B7D28E6F10198B1574FDD03A9053C08A1854A168AA5A57470EC97DD5CE090124EF52A2F7ECBFFD3",
            "00C328FAFCBD79DD77850370C46325D987CB525569FB63C5D3BC53950E6D4C5F174E25A1EE9017B5D450606ADD152B534931D7D4E8455CC91F9B15BF05EC36E377FA",
            "00617CCE7CF5064806C467F678D3B4080D6F1CC50AF26CA209417308281B68AF282623EAA63E5B5C0723D8B8C37FF0777B1A20F8CCB1DCCC43997F1EE0E44DA4A67A",
            SHA512
        ),
        (
            "test",
            "00BB9F2BF4FE1038CCF4DABD7139A56F6FD8BB1386561BD3C6A4FC818B20DF5DDBA80795A947107A1AB9D12DAA615B1ADE4F7A9DC05E8E6311150F47F5C57CE8B222",
            "013BAD9F29ABE20DE37EBEB823C252CA0F63361284015A3BF430A46AAA80B87B0693F0694BD88AFE4E661FC33B094CD3B7963BED5A727ED8BD6A3A202ABE009D0367",
            "01E9BB81FF7944CA409AD138DBBEE228E1AFCC0C890FC78EC8604639CB0DBDC90F717A99EAD9D272855D00162EE9527567DD6A92CBD629805C0445282BBC916797FF",
            SHA1
        ),
        (
            "test",
            "0040D09FCF3C8A5F62CF4FB223CBBB2B9937F6B0577C27020A99602C25A01136987E452988781484EDBBCF1C47E554E7FC901BC3085E5206D9F619CFF07E73D6F706",
            "01C7ED902E123E6815546065A2C4AF977B22AA8EADDB68B2C1110E7EA44D42086BFE4A34B67DDC0E17E96536E358219B23A706C6A6E16BA77B65E1C595D43CAE17FB",
            "0177336676304FCB343CE028B38E7B4FBA76C1C1B277DA18CAD2A8478B2A9A9F5BEC0F3BA04F35DB3E4263569EC6AADE8C92746E4C82F8299AE1B8F1739F8FD519A4",
            SHA224
        ),
        (
            "test",
            "001DE74955EFAABC4C4F17F8E84D881D1310B5392D7700275F82F145C61E843841AF09035BF7A6210F5A431A6A9E81C9323354A9E69135D44EBD2FCAA7731B909258",
            "000E871C4A14F993C6C7369501900C4BC1E9C7B0B4BA44E04868B30B41D8071042EB28C4C250411D0CE08CD197E4188EA4876F279F90B3D8D74A3C76E6F1E4656AA8",
            "00CD52DBAA33B063C3A6CD8058A1FB0A46A4754B034FCC644766CA14DA8CA5CA9FDE00E88C1AD60CCBA759025299079D7A427EC3CC5B619BFBC828E7769BCD694E86",
            SHA256
        ),
        (
            "test",
            "01F1FC4A349A7DA9A9E116BFDD055DC08E78252FF8E23AC276AC88B1770AE0B5DCEB1ED14A4916B769A523CE1E90BA22846AF11DF8B300C38818F713DADD85DE0C88",
            "014BEE21A18B6D8B3C93FAB08D43E739707953244FDBE924FA926D76669E7AC8C89DF62ED8975C2D8397A65A49DCC09F6B0AC62272741924D479354D74FF6075578C",
            "0133330865C067A0EAF72362A65E2D7BC4E461E8C8995C3B6226A21BD1AA78F0ED94FE536A0DCA35534F0CD1510C41525D163FE9D74D134881E35141ED5E8E95B979",
            SHA384
        ),
        (
            "test",
            "016200813020EC986863BEDFC1B121F605C1215645018AEA1A7B215A564DE9EB1B38A67AA1128B80CE391C4FB71187654AAA3431027BFC7F395766CA988C964DC56D",
            "013E99020ABF5CEE7525D16B69B229652AB6BDF2AFFCAEF38773B4B7D08725F10CDB93482FDCC54EDCEE91ECA4166B2A7C6265EF0CE2BD7051B7CEF945BABD47EE6D",
            "01FBD0013C674AA79CB39849527916CE301C66EA7CE8B80682786AD60F98F7E78A19CA69EFF5C57400E3B3A0AD66CE0978214D13BAF4E9AC60752F7B155E2DE4DCE3",
            SHA512
        ),
    )

    signatures_p192 = []
    for a, b, c, d, e in signatures_p192_:
        new_tv = (tobytes(a), unhexlify(b), unhexlify(c), unhexlify(d), e)
        signatures_p192.append(new_tv)

    signatures_p224 = []
    for a, b, c, d, e in signatures_p224_:
        new_tv = (tobytes(a), unhexlify(b), unhexlify(c), unhexlify(d), e)
        signatures_p224.append(new_tv)

    signatures_p256 = []
    for a, b, c, d, e in signatures_p256_:
        new_tv = (tobytes(a), unhexlify(b), unhexlify(c), unhexlify(d), e)
        signatures_p256.append(new_tv)

    signatures_p384 = []
    for a, b, c, d, e in signatures_p384_:
        new_tv = (tobytes(a), unhexlify(b), unhexlify(c), unhexlify(d), e)
        signatures_p384.append(new_tv)

    signatures_p521 = []
    for a, b, c, d, e in signatures_p521_:
        new_tv = (tobytes(a), unhexlify(b), unhexlify(c), unhexlify(d), e)
        signatures_p521.append(new_tv)

    def shortDescription(self):
        return "Deterministic ECDSA Tests"

    def test_loopback_p192(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv_p192, 'deterministic-rfc6979')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub_p192, 'deterministic-rfc6979')
        verifier.verify(hashed_msg, signature)

    def test_loopback_p224(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv_p224, 'deterministic-rfc6979')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub_p224, 'deterministic-rfc6979')
        verifier.verify(hashed_msg, signature)

    def test_loopback_p256(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv_p256, 'deterministic-rfc6979')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub_p256, 'deterministic-rfc6979')
        verifier.verify(hashed_msg, signature)

    def test_loopback_p384(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv_p384, 'deterministic-rfc6979')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub_p384, 'deterministic-rfc6979')
        verifier.verify(hashed_msg, signature)

    def test_loopback_p521(self):
        hashed_msg = SHA512.new(b"test")
        signer = DSS.new(self.key_priv_p521, 'deterministic-rfc6979')
        signature = signer.sign(hashed_msg)

        verifier = DSS.new(self.key_pub_p521, 'deterministic-rfc6979')
        verifier.verify(hashed_msg, signature)

    def test_data_rfc6979_p192(self):
        signer = DSS.new(self.key_priv_p192, 'deterministic-rfc6979')
        for message, k, r, s, module in self.signatures_p192:
            hash_obj = module.new(message)
            result = signer.sign(hash_obj)
            self.assertEqual(r + s, result)

    def test_data_rfc6979_p224(self):
        signer = DSS.new(self.key_priv_p224, 'deterministic-rfc6979')
        for message, k, r, s, module in self.signatures_p224:
            hash_obj = module.new(message)
            result = signer.sign(hash_obj)
            self.assertEqual(r + s, result)

    def test_data_rfc6979_p256(self):
        signer = DSS.new(self.key_priv_p256, 'deterministic-rfc6979')
        for message, k, r, s, module in self.signatures_p256:
            hash_obj = module.new(message)
            result = signer.sign(hash_obj)
            self.assertEqual(r + s, result)

    def test_data_rfc6979_p384(self):
        signer = DSS.new(self.key_priv_p384, 'deterministic-rfc6979')
        for message, k, r, s, module in self.signatures_p384:
            hash_obj = module.new(message)
            result = signer.sign(hash_obj)
            self.assertEqual(r + s, result)

    def test_data_rfc6979_p521(self):
        signer = DSS.new(self.key_priv_p521, 'deterministic-rfc6979')
        for message, k, r, s, module in self.signatures_p521:
            hash_obj = module.new(message)
            result = signer.sign(hash_obj)
            self.assertEqual(r + s, result)


def get_hash_module(hash_name):
    if hash_name == "SHA-512":
        hash_module = SHA512
    elif hash_name == "SHA-512/224":
        hash_module = SHA512.new(truncate="224")
    elif hash_name == "SHA-512/256":
        hash_module = SHA512.new(truncate="256")
    elif hash_name == "SHA-384":
        hash_module = SHA384
    elif hash_name == "SHA-256":
        hash_module = SHA256
    elif hash_name == "SHA-224":
        hash_module = SHA224
    elif hash_name == "SHA-1":
        hash_module = SHA1
    elif hash_name == "SHA3-224":
        hash_module = SHA3_224
    elif hash_name == "SHA3-256":
        hash_module = SHA3_256
    elif hash_name == "SHA3-384":
        hash_module = SHA3_384
    elif hash_name == "SHA3-512":
        hash_module = SHA3_512
    else:
        raise ValueError("Unknown hash algorithm: " + hash_name)
    return hash_module


class TestVectorsDSAWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, slow_tests):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._slow_tests = slow_tests
        self._id = "None"
        self.tv = []

    def setUp(self):

        def filter_dsa(group):
            return DSA.import_key(group['keyPem'])

        def filter_sha(group):
            return get_hash_module(group['sha'])

        def filter_type(group):
            sig_type = group['type']
            if sig_type != 'DsaVerify':
                raise ValueError("Unknown signature type " + sig_type)
            return sig_type

        result = load_test_vectors_wycheproof(("Signature", "wycheproof"),
                                              "dsa_test.json",
                                              "Wycheproof DSA signature",
                                              group_tag={'key': filter_dsa,
                                                         'hash_module': filter_sha,
                                                         'sig_type': filter_type})
        self.tv += result

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = "Wycheproof DSA Test #" + str(tv.id)

        hashed_msg = tv.hash_module.new(tv.msg)
        signer = DSS.new(tv.key, 'fips-186-3', encoding='der')
        try:
            signature = signer.verify(hashed_msg, tv.sig)
        except ValueError as e:
            if tv.warning:
                return
            assert not tv.valid
        else:
            assert tv.valid
            self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)


class TestVectorsECDSAWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, slow_tests):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._slow_tests = slow_tests
        self._id = "None"

    def add_tests(self, filename):

        def filter_ecc(group):
            # These are the only curves we accept to skip
            if group['key']['curve'] in ('secp224k1', 'secp256k1',
                                         'brainpoolP224r1', 'brainpoolP224t1',
                                         'brainpoolP256r1', 'brainpoolP256t1',
                                         'brainpoolP320r1', 'brainpoolP320t1',
                                         'brainpoolP384r1', 'brainpoolP384t1',
                                         'brainpoolP512r1', 'brainpoolP512t1',
                                         ):
                return None
            return ECC.import_key(group['keyPem'])

        def filter_sha(group):
            return get_hash_module(group['sha'])

        def filter_encoding(group):
            encoding_name = group['type']
            if encoding_name == "EcdsaVerify":
                return "der"
            elif encoding_name == "EcdsaP1363Verify":
                return "binary"
            else:
                raise ValueError("Unknown signature type " + encoding_name)

        result = load_test_vectors_wycheproof(("Signature", "wycheproof"),
                                              filename,
                                              "Wycheproof ECDSA signature (%s)" % filename,
                                              group_tag={'key': filter_ecc,
                                                         'hash_module': filter_sha,
                                                         'encoding': filter_encoding,
                                                         })
        self.tv += result

    def setUp(self):
        self.tv = []
        self.add_tests("ecdsa_secp224r1_sha224_p1363_test.json")
        self.add_tests("ecdsa_secp224r1_sha224_test.json")
        if self._slow_tests:
            self.add_tests("ecdsa_secp224r1_sha256_p1363_test.json")
            self.add_tests("ecdsa_secp224r1_sha256_test.json")
            self.add_tests("ecdsa_secp224r1_sha3_224_test.json")
            self.add_tests("ecdsa_secp224r1_sha3_256_test.json")
            self.add_tests("ecdsa_secp224r1_sha3_512_test.json")
            self.add_tests("ecdsa_secp224r1_sha512_p1363_test.json")
            self.add_tests("ecdsa_secp224r1_sha512_test.json")
            self.add_tests("ecdsa_secp256r1_sha256_p1363_test.json")
            self.add_tests("ecdsa_secp256r1_sha256_test.json")
            self.add_tests("ecdsa_secp256r1_sha3_256_test.json")
            self.add_tests("ecdsa_secp256r1_sha3_512_test.json")
            self.add_tests("ecdsa_secp256r1_sha512_p1363_test.json")
        self.add_tests("ecdsa_secp256r1_sha512_test.json")
        if self._slow_tests:
            self.add_tests("ecdsa_secp384r1_sha3_384_test.json")
            self.add_tests("ecdsa_secp384r1_sha3_512_test.json")
            self.add_tests("ecdsa_secp384r1_sha384_p1363_test.json")
            self.add_tests("ecdsa_secp384r1_sha384_test.json")
            self.add_tests("ecdsa_secp384r1_sha512_p1363_test.json")
        self.add_tests("ecdsa_secp384r1_sha512_test.json")
        if self._slow_tests:
            self.add_tests("ecdsa_secp521r1_sha3_512_test.json")
            self.add_tests("ecdsa_secp521r1_sha512_p1363_test.json")
        self.add_tests("ecdsa_secp521r1_sha512_test.json")
        self.add_tests("ecdsa_test.json")
        self.add_tests("ecdsa_webcrypto_test.json")

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = "Wycheproof ECDSA Test #%d (%s, %s)" % (tv.id, tv.comment, tv.filename)

        # Skip tests with unsupported curves
        if tv.key is None:
            return

        hashed_msg = tv.hash_module.new(tv.msg)
        signer = DSS.new(tv.key, 'fips-186-3', encoding=tv.encoding)
        try:
            signature = signer.verify(hashed_msg, tv.sig)
        except ValueError as e:
            if tv.warning:
                return
            if tv.comment == "k*G has a large x-coordinate":
                return
            assert not tv.valid
        else:
            assert tv.valid
            self.warn(tv)

    def runTest(self):
        for tv in self.tv:
            self.test_verify(tv)


def get_tests(config={}):
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(FIPS_DSA_Tests)
    tests += list_test_cases(FIPS_ECDSA_Tests)
    tests += list_test_cases(Det_DSA_Tests)
    tests += list_test_cases(Det_ECDSA_Tests)

    slow_tests = config.get('slow_tests')
    if slow_tests:
        tests += list_test_cases(FIPS_DSA_Tests_KAT)
        tests += list_test_cases(FIPS_ECDSA_Tests_KAT)

    tests += [TestVectorsDSAWycheproof(wycheproof_warnings, slow_tests)]
    tests += [TestVectorsECDSAWycheproof(wycheproof_warnings, slow_tests)]

    return tests


if __name__ == '__main__':
    def suite():
        return unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
