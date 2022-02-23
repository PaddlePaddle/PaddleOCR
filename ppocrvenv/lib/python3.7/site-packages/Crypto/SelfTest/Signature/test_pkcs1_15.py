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

import json
import unittest
from binascii import unhexlify

from Crypto.Util.py3compat import bchr
from Crypto.Util.number import bytes_to_long
from Crypto.Util.strxor import strxor
from Crypto.SelfTest.st_common import list_test_cases
from Crypto.SelfTest.loader import load_test_vectors, load_test_vectors_wycheproof

from Crypto.Hash import (SHA1, SHA224, SHA256, SHA384, SHA512, SHA3_384,
                         SHA3_224, SHA3_256, SHA3_512)
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Signature import PKCS1_v1_5

from Crypto.Util._file_system import pycryptodome_filename
from Crypto.Util.strxor import strxor


def load_hash_by_name(hash_name):
    return __import__("Crypto.Hash." + hash_name, globals(), locals(), ["new"])


class FIPS_PKCS1_Verify_Tests(unittest.TestCase):

    def shortDescription(self):
        return "FIPS PKCS1 Tests (Verify)"

    def test_can_sign(self):
        test_public_key = RSA.generate(1024).public_key()
        verifier = pkcs1_15.new(test_public_key)
        self.assertEqual(verifier.can_sign(), False)


class FIPS_PKCS1_Verify_Tests_KAT(unittest.TestCase):
    pass


test_vectors_verify = load_test_vectors(("Signature", "PKCS1-v1.5"),
                                 "SigVer15_186-3.rsp",
                                 "Signature Verification 186-3",
                                 {'shaalg': lambda x: x,
                                  'd': lambda x: int(x),
                                  'result': lambda x: x}) or []


for count, tv in enumerate(test_vectors_verify):
    if isinstance(tv, str):
        continue
    if hasattr(tv, "n"):
        modulus = tv.n
        continue

    hash_module = load_hash_by_name(tv.shaalg.upper())
    hash_obj = hash_module.new(tv.msg)
    public_key = RSA.construct([bytes_to_long(x) for x in (modulus, tv.e)]) # type: ignore
    verifier = pkcs1_15.new(public_key)

    def positive_test(self, hash_obj=hash_obj, verifier=verifier, signature=tv.s):
        verifier.verify(hash_obj, signature)

    def negative_test(self, hash_obj=hash_obj, verifier=verifier, signature=tv.s):
        self.assertRaises(ValueError, verifier.verify, hash_obj, signature)

    if tv.result == 'f':
        setattr(FIPS_PKCS1_Verify_Tests_KAT, "test_negative_%d" % count, negative_test)
    else:
        setattr(FIPS_PKCS1_Verify_Tests_KAT, "test_positive_%d" % count, positive_test)


class FIPS_PKCS1_Sign_Tests(unittest.TestCase):

    def shortDescription(self):
        return "FIPS PKCS1 Tests (Sign)"

    def test_can_sign(self):
        test_private_key = RSA.generate(1024)
        signer = pkcs1_15.new(test_private_key)
        self.assertEqual(signer.can_sign(), True)


class FIPS_PKCS1_Sign_Tests_KAT(unittest.TestCase):
    pass


test_vectors_sign  = load_test_vectors(("Signature", "PKCS1-v1.5"),
                                        "SigGen15_186-2.txt",
                                        "Signature Generation 186-2",
                                        {'shaalg': lambda x: x}) or []

test_vectors_sign += load_test_vectors(("Signature", "PKCS1-v1.5"),
                                        "SigGen15_186-3.txt",
                                        "Signature Generation 186-3",
                                        {'shaalg': lambda x: x}) or []

for count, tv in enumerate(test_vectors_sign):
    if isinstance(tv, str):
        continue
    if hasattr(tv, "n"):
        modulus = tv.n
        continue
    if hasattr(tv, "e"):
        private_key = RSA.construct([bytes_to_long(x) for x in (modulus, tv.e, tv.d)]) # type: ignore
        signer = pkcs1_15.new(private_key)
        continue

    hash_module = load_hash_by_name(tv.shaalg.upper())
    hash_obj = hash_module.new(tv.msg)

    def new_test(self, hash_obj=hash_obj, signer=signer, result=tv.s):
        signature = signer.sign(hash_obj)
        self.assertEqual(signature, result)

    setattr(FIPS_PKCS1_Sign_Tests_KAT, "test_%d" % count, new_test)


class PKCS1_15_NoParams(unittest.TestCase):
    """Verify that PKCS#1 v1.5 signatures pass even without NULL parameters in
    the algorithm identifier (PyCrypto/LP bug #1119552)."""

    rsakey = """-----BEGIN RSA PRIVATE KEY-----
            MIIBOwIBAAJBAL8eJ5AKoIsjURpcEoGubZMxLD7+kT+TLr7UkvEtFrRhDDKMtuII
            q19FrL4pUIMymPMSLBn3hJLe30Dw48GQM4UCAwEAAQJACUSDEp8RTe32ftq8IwG8
            Wojl5mAd1wFiIOrZ/Uv8b963WJOJiuQcVN29vxU5+My9GPZ7RA3hrDBEAoHUDPrI
            OQIhAPIPLz4dphiD9imAkivY31Rc5AfHJiQRA7XixTcjEkojAiEAyh/pJHks/Mlr
            +rdPNEpotBjfV4M4BkgGAA/ipcmaAjcCIQCHvhwwKVBLzzTscT2HeUdEeBMoiXXK
            JACAr3sJQJGxIQIgarRp+m1WSKV1MciwMaTOnbU7wxFs9DP1pva76lYBzgUCIQC9
            n0CnZCJ6IZYqSt0H5N7+Q+2Ro64nuwV/OSQfM6sBwQ==
            -----END RSA PRIVATE KEY-----"""

    msg = b"This is a test\x0a"

    # PKCS1 v1.5 signature of the message computed using SHA-1.
    # The digestAlgorithm SEQUENCE does NOT contain the NULL parameter.
    sig_str = "a287a13517f716e72fb14eea8e33a8db4a4643314607e7ca3e3e28"\
              "1893db74013dda8b855fd99f6fecedcb25fcb7a434f35cd0a101f8"\
              "b19348e0bd7b6f152dfc"
    signature = unhexlify(sig_str)

    def runTest(self):
        verifier = pkcs1_15.new(RSA.importKey(self.rsakey))
        hashed = SHA1.new(self.msg)
        verifier.verify(hashed, self.signature)


class PKCS1_Legacy_Module_Tests(unittest.TestCase):
    """Verify that the legacy module Crypto.Signature.PKCS1_v1_5
    behaves as expected. The only difference is that the verify()
    method returns True/False and does not raise exceptions."""

    def shortDescription(self):
        return "Test legacy Crypto.Signature.PKCS1_v1_5"

    def runTest(self):
        key = RSA.importKey(PKCS1_15_NoParams.rsakey)
        hashed = SHA1.new(b"Test")
        good_signature = PKCS1_v1_5.new(key).sign(hashed)
        verifier = PKCS1_v1_5.new(key.public_key())

        self.assertEqual(verifier.verify(hashed, good_signature), True)

        # Flip a few bits in the signature
        bad_signature = strxor(good_signature, bchr(1) * len(good_signature))
        self.assertEqual(verifier.verify(hashed, bad_signature), False)


class PKCS1_All_Hashes_Tests(unittest.TestCase):

    def shortDescription(self):
        return "Test PKCS#1v1.5 signature in combination with all hashes"

    def runTest(self):

        key = RSA.generate(1024)
        signer = pkcs1_15.new(key)
        hash_names = ("MD2", "MD4", "MD5", "RIPEMD160", "SHA1",
                      "SHA224", "SHA256", "SHA384", "SHA512",
                      "SHA3_224", "SHA3_256", "SHA3_384", "SHA3_512")

        for name in hash_names:
            hashed = load_hash_by_name(name).new(b"Test")
            signer.sign(hashed)

        from Crypto.Hash import BLAKE2b, BLAKE2s
        for hash_size in (20, 32, 48, 64):
            hashed_b = BLAKE2b.new(digest_bytes=hash_size, data=b"Test")
            signer.sign(hashed_b)
        for hash_size in (16, 20, 28, 32):
            hashed_s = BLAKE2s.new(digest_bytes=hash_size, data=b"Test")
            signer.sign(hashed_s)


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._id = "None"

    def setUp(self):
        self.tv = []
        self.add_tests("rsa_sig_gen_misc_test.json")
        self.add_tests("rsa_signature_2048_sha224_test.json")
        self.add_tests("rsa_signature_2048_sha256_test.json")
        self.add_tests("rsa_signature_2048_sha384_test.json")
        self.add_tests("rsa_signature_2048_sha3_224_test.json")
        self.add_tests("rsa_signature_2048_sha3_256_test.json")
        self.add_tests("rsa_signature_2048_sha3_384_test.json")
        self.add_tests("rsa_signature_2048_sha3_512_test.json")
        self.add_tests("rsa_signature_2048_sha512_test.json")
        self.add_tests("rsa_signature_2048_sha512_224_test.json")
        self.add_tests("rsa_signature_2048_sha512_256_test.json")
        self.add_tests("rsa_signature_3072_sha256_test.json")
        self.add_tests("rsa_signature_3072_sha384_test.json")
        self.add_tests("rsa_signature_3072_sha3_256_test.json")
        self.add_tests("rsa_signature_3072_sha3_384_test.json")
        self.add_tests("rsa_signature_3072_sha3_512_test.json")
        self.add_tests("rsa_signature_3072_sha512_test.json")
        self.add_tests("rsa_signature_3072_sha512_256_test.json")
        self.add_tests("rsa_signature_4096_sha384_test.json")
        self.add_tests("rsa_signature_4096_sha512_test.json")
        self.add_tests("rsa_signature_4096_sha512_256_test.json")
        self.add_tests("rsa_signature_test.json")

    def add_tests(self, filename):

        def filter_rsa(group):
            return RSA.import_key(group['keyPem'])

        def filter_sha(group):
            hash_name = group['sha']
            if hash_name == "SHA-512":
                return SHA512
            elif hash_name == "SHA-512/224":
                return SHA512.new(truncate="224")
            elif hash_name == "SHA-512/256":
                return SHA512.new(truncate="256")
            elif hash_name == "SHA3-512":
                return SHA3_512
            elif hash_name == "SHA-384":
                return SHA384
            elif hash_name == "SHA3-384":
                return SHA3_384
            elif hash_name == "SHA-256":
                return SHA256
            elif hash_name == "SHA3-256":
                return SHA3_256
            elif hash_name == "SHA-224":
                return SHA224
            elif hash_name == "SHA3-224":
                return SHA3_224
            elif hash_name == "SHA-1":
                return SHA1
            else:
                raise ValueError("Unknown hash algorithm: " + hash_name)

        def filter_type(group):
            type_name = group['type']
            if type_name not in ("RsassaPkcs1Verify", "RsassaPkcs1Generate"):
                raise ValueError("Unknown type name " + type_name)

        result = load_test_vectors_wycheproof(("Signature", "wycheproof"),
                                              filename,
                                              "Wycheproof PKCS#1v1.5 signature (%s)" % filename,
                                              group_tag={'rsa_key': filter_rsa,
                                                         'hash_mod': filter_sha,
                                                         'type': filter_type})
        return result

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_verify(self, tv):
        self._id = "Wycheproof RSA PKCS$#1 Test #" + str(tv.id)

        hashed_msg = tv.hash_module.new(tv.msg)
        signer = pkcs1_15.new(tv.key)
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


def get_tests(config={}):
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(FIPS_PKCS1_Verify_Tests)
    tests += list_test_cases(FIPS_PKCS1_Sign_Tests)
    tests += list_test_cases(PKCS1_15_NoParams)
    tests += list_test_cases(PKCS1_Legacy_Module_Tests)
    tests += list_test_cases(PKCS1_All_Hashes_Tests)
    tests += [ TestVectorsWycheproof(wycheproof_warnings) ]

    if config.get('slow_tests'):
        tests += list_test_cases(FIPS_PKCS1_Verify_Tests_KAT)
        tests += list_test_cases(FIPS_PKCS1_Sign_Tests_KAT)

    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')
