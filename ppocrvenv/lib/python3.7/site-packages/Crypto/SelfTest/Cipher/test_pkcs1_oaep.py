# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/test_pkcs1_oaep.py: Self-test for PKCS#1 OAEP encryption
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

from Crypto.SelfTest.st_common import list_test_cases, a2b_hex
from Crypto.SelfTest.loader import load_test_vectors_wycheproof

from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP as PKCS
from Crypto.Hash import MD2, MD5, SHA1, SHA256, RIPEMD160, SHA224, SHA384, SHA512
from Crypto import Random
from Crypto.Signature.pss import MGF1

from Crypto.Util.py3compat import b, bchr


def rws(t):
    """Remove white spaces, tabs, and new lines from a string"""
    for c in ['\n', '\t', ' ']:
        t = t.replace(c, '')
    return t


def t2b(t):
    """Convert a text string with bytes in hex form to a byte string"""
    clean = rws(t)
    if len(clean) % 2 == 1:
        raise ValueError("Even number of characters expected")
    return a2b_hex(clean)


class PKCS1_OAEP_Tests(unittest.TestCase):

        def setUp(self):
                self.rng = Random.new().read
                self.key1024 = RSA.generate(1024, self.rng)

        # List of tuples with test data for PKCS#1 OAEP
        # Each tuple is made up by:
        #       Item #0: dictionary with RSA key component
        #       Item #1: plaintext
        #       Item #2: ciphertext
        #       Item #3: random data (=seed)
        #       Item #4: hash object

        _testData = (

                #
                # From in oaep-int.txt to be found in
                # ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
                #
                (
                # Private key
                {
                'n':'''bb f8 2f 09 06 82 ce 9c 23 38 ac 2b 9d a8 71 f7
                36 8d 07 ee d4 10 43 a4 40 d6 b6 f0 74 54 f5 1f
                b8 df ba af 03 5c 02 ab 61 ea 48 ce eb 6f cd 48
                76 ed 52 0d 60 e1 ec 46 19 71 9d 8a 5b 8b 80 7f
                af b8 e0 a3 df c7 37 72 3e e6 b4 b7 d9 3a 25 84
                ee 6a 64 9d 06 09 53 74 88 34 b2 45 45 98 39 4e
                e0 aa b1 2d 7b 61 a5 1f 52 7a 9a 41 f6 c1 68 7f
                e2 53 72 98 ca 2a 8f 59 46 f8 e5 fd 09 1d bd cb''',
                # Public key
                'e':'11',
                # In the test vector, only p and q were given...
                # d is computed offline as e^{-1} mod (p-1)(q-1)
                'd':'''a5dafc5341faf289c4b988db30c1cdf83f31251e0
                668b42784813801579641b29410b3c7998d6bc465745e5c3
                92669d6870da2c082a939e37fdcb82ec93edac97ff3ad595
                0accfbc111c76f1a9529444e56aaf68c56c092cd38dc3bef
                5d20a939926ed4f74a13eddfbe1a1cecc4894af9428c2b7b
                8883fe4463a4bc85b1cb3c1'''
                }
                ,
                # Plaintext
                '''d4 36 e9 95 69 fd 32 a7 c8 a0 5b bc 90 d3 2c 49''',
                # Ciphertext
                '''12 53 e0 4d c0 a5 39 7b b4 4a 7a b8 7e 9b f2 a0
                39 a3 3d 1e 99 6f c8 2a 94 cc d3 00 74 c9 5d f7
                63 72 20 17 06 9e 52 68 da 5d 1c 0b 4f 87 2c f6
                53 c1 1d f8 23 14 a6 79 68 df ea e2 8d ef 04 bb
                6d 84 b1 c3 1d 65 4a 19 70 e5 78 3b d6 eb 96 a0
                24 c2 ca 2f 4a 90 fe 9f 2e f5 c9 c1 40 e5 bb 48
                da 95 36 ad 87 00 c8 4f c9 13 0a de a7 4e 55 8d
                51 a7 4d df 85 d8 b5 0d e9 68 38 d6 06 3e 09 55''',
                # Random
                '''aa fd 12 f6 59 ca e6 34 89 b4 79 e5 07 6d de c2
                f0 6c b5 8f''',
                # Hash
                SHA1,
               ),

                #
                # From in oaep-vect.txt to be found in Example 1.1
                # ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
                #
                (
                # Private key
                {
                'n':'''a8 b3 b2 84 af 8e b5 0b 38 70 34 a8 60 f1 46 c4
                91 9f 31 87 63 cd 6c 55 98 c8 ae 48 11 a1 e0 ab
                c4 c7 e0 b0 82 d6 93 a5 e7 fc ed 67 5c f4 66 85
                12 77 2c 0c bc 64 a7 42 c6 c6 30 f5 33 c8 cc 72
                f6 2a e8 33 c4 0b f2 58 42 e9 84 bb 78 bd bf 97
                c0 10 7d 55 bd b6 62 f5 c4 e0 fa b9 84 5c b5 14
                8e f7 39 2d d3 aa ff 93 ae 1e 6b 66 7b b3 d4 24
                76 16 d4 f5 ba 10 d4 cf d2 26 de 88 d3 9f 16 fb''',
                'e':'''01 00 01''',
                'd':'''53 33 9c fd b7 9f c8 46 6a 65 5c 73 16 ac a8 5c
                55 fd 8f 6d d8 98 fd af 11 95 17 ef 4f 52 e8 fd
                8e 25 8d f9 3f ee 18 0f a0 e4 ab 29 69 3c d8 3b
                15 2a 55 3d 4a c4 d1 81 2b 8b 9f a5 af 0e 7f 55
                fe 73 04 df 41 57 09 26 f3 31 1f 15 c4 d6 5a 73
                2c 48 31 16 ee 3d 3d 2d 0a f3 54 9a d9 bf 7c bf
                b7 8a d8 84 f8 4d 5b eb 04 72 4d c7 36 9b 31 de
                f3 7d 0c f5 39 e9 cf cd d3 de 65 37 29 ea d5 d1 '''
                }
                ,
                # Plaintext
                '''66 28 19 4e 12 07 3d b0 3b a9 4c da 9e f9 53 23
                97 d5 0d ba 79 b9 87 00 4a fe fe 34''',
                # Ciphertext
                '''35 4f e6 7b 4a 12 6d 5d 35 fe 36 c7 77 79 1a 3f
                7b a1 3d ef 48 4e 2d 39 08 af f7 22 fa d4 68 fb
                21 69 6d e9 5d 0b e9 11 c2 d3 17 4f 8a fc c2 01
                03 5f 7b 6d 8e 69 40 2d e5 45 16 18 c2 1a 53 5f
                a9 d7 bf c5 b8 dd 9f c2 43 f8 cf 92 7d b3 13 22
                d6 e8 81 ea a9 1a 99 61 70 e6 57 a0 5a 26 64 26
                d9 8c 88 00 3f 84 77 c1 22 70 94 a0 d9 fa 1e 8c
                40 24 30 9c e1 ec cc b5 21 00 35 d4 7a c7 2e 8a''',
                # Random
                '''18 b7 76 ea 21 06 9d 69 77 6a 33 e9 6b ad 48 e1
                dd a0 a5 ef''',
                SHA1
                ),

                #
                # From in oaep-vect.txt to be found in Example 2.1
                # ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
                #
                (
                # Private key
                {
                'n':'''01 94 7c 7f ce 90 42 5f 47 27 9e 70 85 1f 25 d5
                e6 23 16 fe 8a 1d f1 93 71 e3 e6 28 e2 60 54 3e
                49 01 ef 60 81 f6 8c 0b 81 41 19 0d 2a e8 da ba
                7d 12 50 ec 6d b6 36 e9 44 ec 37 22 87 7c 7c 1d
                0a 67 f1 4b 16 94 c5 f0 37 94 51 a4 3e 49 a3 2d
                de 83 67 0b 73 da 91 a1 c9 9b c2 3b 43 6a 60 05
                5c 61 0f 0b af 99 c1 a0 79 56 5b 95 a3 f1 52 66
                32 d1 d4 da 60 f2 0e da 25 e6 53 c4 f0 02 76 6f
                45''',
                'e':'''01 00 01''',
                'd':'''08 23 f2 0f ad b5 da 89 08 8a 9d 00 89 3e 21 fa
                4a 1b 11 fb c9 3c 64 a3 be 0b aa ea 97 fb 3b 93
                c3 ff 71 37 04 c1 9c 96 3c 1d 10 7a ae 99 05 47
                39 f7 9e 02 e1 86 de 86 f8 7a 6d de fe a6 d8 cc
                d1 d3 c8 1a 47 bf a7 25 5b e2 06 01 a4 a4 b2 f0
                8a 16 7b 5e 27 9d 71 5b 1b 45 5b dd 7e ab 24 59
                41 d9 76 8b 9a ce fb 3c cd a5 95 2d a3 ce e7 25
                25 b4 50 16 63 a8 ee 15 c9 e9 92 d9 24 62 fe 39'''
                },
                # Plaintext
                '''8f f0 0c aa 60 5c 70 28 30 63 4d 9a 6c 3d 42 c6
                52 b5 8c f1 d9 2f ec 57 0b ee e7''',
                # Ciphertext
                '''01 81 af 89 22 b9 fc b4 d7 9d 92 eb e1 98 15 99
                2f c0 c1 43 9d 8b cd 49 13 98 a0 f4 ad 3a 32 9a
                5b d9 38 55 60 db 53 26 83 c8 b7 da 04 e4 b1 2a
                ed 6a ac df 47 1c 34 c9 cd a8 91 ad dc c2 df 34
                56 65 3a a6 38 2e 9a e5 9b 54 45 52 57 eb 09 9d
                56 2b be 10 45 3f 2b 6d 13 c5 9c 02 e1 0f 1f 8a
                bb 5d a0 d0 57 09 32 da cf 2d 09 01 db 72 9d 0f
                ef cc 05 4e 70 96 8e a5 40 c8 1b 04 bc ae fe 72
                0e''',
                # Random
                '''8c 40 7b 5e c2 89 9e 50 99 c5 3e 8c e7 93 bf 94
                e7 1b 17 82''',
                SHA1
                ),

                #
                # From in oaep-vect.txt to be found in Example 10.1
                # ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1-vec.zip
                #
                (
                # Private key
                {
                'n':'''ae 45 ed 56 01 ce c6 b8 cc 05 f8 03 93 5c 67 4d
                db e0 d7 5c 4c 09 fd 79 51 fc 6b 0c ae c3 13 a8
                df 39 97 0c 51 8b ff ba 5e d6 8f 3f 0d 7f 22 a4
                02 9d 41 3f 1a e0 7e 4e be 9e 41 77 ce 23 e7 f5
                40 4b 56 9e 4e e1 bd cf 3c 1f b0 3e f1 13 80 2d
                4f 85 5e b9 b5 13 4b 5a 7c 80 85 ad ca e6 fa 2f
                a1 41 7e c3 76 3b e1 71 b0 c6 2b 76 0e de 23 c1
                2a d9 2b 98 08 84 c6 41 f5 a8 fa c2 6b da d4 a0
                33 81 a2 2f e1 b7 54 88 50 94 c8 25 06 d4 01 9a
                53 5a 28 6a fe b2 71 bb 9b a5 92 de 18 dc f6 00
                c2 ae ea e5 6e 02 f7 cf 79 fc 14 cf 3b dc 7c d8
                4f eb bb f9 50 ca 90 30 4b 22 19 a7 aa 06 3a ef
                a2 c3 c1 98 0e 56 0c d6 4a fe 77 95 85 b6 10 76
                57 b9 57 85 7e fd e6 01 09 88 ab 7d e4 17 fc 88
                d8 f3 84 c4 e6 e7 2c 3f 94 3e 0c 31 c0 c4 a5 cc
                36 f8 79 d8 a3 ac 9d 7d 59 86 0e aa da 6b 83 bb''',
                'e':'''01 00 01''',
                'd':'''05 6b 04 21 6f e5 f3 54 ac 77 25 0a 4b 6b 0c 85
                25 a8 5c 59 b0 bd 80 c5 64 50 a2 2d 5f 43 8e 59
                6a 33 3a a8 75 e2 91 dd 43 f4 8c b8 8b 9d 5f c0
                d4 99 f9 fc d1 c3 97 f9 af c0 70 cd 9e 39 8c 8d
                19 e6 1d b7 c7 41 0a 6b 26 75 df bf 5d 34 5b 80
                4d 20 1a dd 50 2d 5c e2 df cb 09 1c e9 99 7b be
                be 57 30 6f 38 3e 4d 58 81 03 f0 36 f7 e8 5d 19
                34 d1 52 a3 23 e4 a8 db 45 1d 6f 4a 5b 1b 0f 10
                2c c1 50 e0 2f ee e2 b8 8d ea 4a d4 c1 ba cc b2
                4d 84 07 2d 14 e1 d2 4a 67 71 f7 40 8e e3 05 64
                fb 86 d4 39 3a 34 bc f0 b7 88 50 1d 19 33 03 f1
                3a 22 84 b0 01 f0 f6 49 ea f7 93 28 d4 ac 5c 43
                0a b4 41 49 20 a9 46 0e d1 b7 bc 40 ec 65 3e 87
                6d 09 ab c5 09 ae 45 b5 25 19 01 16 a0 c2 61 01
                84 82 98 50 9c 1c 3b f3 a4 83 e7 27 40 54 e1 5e
                97 07 50 36 e9 89 f6 09 32 80 7b 52 57 75 1e 79'''
                },
                # Plaintext
                '''8b ba 6b f8 2a 6c 0f 86 d5 f1 75 6e 97 95 68 70
                b0 89 53 b0 6b 4e b2 05 bc 16 94 ee''',
                # Ciphertext
                '''53 ea 5d c0 8c d2 60 fb 3b 85 85 67 28 7f a9 15
                52 c3 0b 2f eb fb a2 13 f0 ae 87 70 2d 06 8d 19
                ba b0 7f e5 74 52 3d fb 42 13 9d 68 c3 c5 af ee
                e0 bf e4 cb 79 69 cb f3 82 b8 04 d6 e6 13 96 14
                4e 2d 0e 60 74 1f 89 93 c3 01 4b 58 b9 b1 95 7a
                8b ab cd 23 af 85 4f 4c 35 6f b1 66 2a a7 2b fc
                c7 e5 86 55 9d c4 28 0d 16 0c 12 67 85 a7 23 eb
                ee be ff 71 f1 15 94 44 0a ae f8 7d 10 79 3a 87
                74 a2 39 d4 a0 4c 87 fe 14 67 b9 da f8 52 08 ec
                6c 72 55 79 4a 96 cc 29 14 2f 9a 8b d4 18 e3 c1
                fd 67 34 4b 0c d0 82 9d f3 b2 be c6 02 53 19 62
                93 c6 b3 4d 3f 75 d3 2f 21 3d d4 5c 62 73 d5 05
                ad f4 cc ed 10 57 cb 75 8f c2 6a ee fa 44 12 55
                ed 4e 64 c1 99 ee 07 5e 7f 16 64 61 82 fd b4 64
                73 9b 68 ab 5d af f0 e6 3e 95 52 01 68 24 f0 54
                bf 4d 3c 8c 90 a9 7b b6 b6 55 32 84 eb 42 9f cc''',
                # Random
                '''47 e1 ab 71 19 fe e5 6c 95 ee 5e aa d8 6f 40 d0
                aa 63 bd 33''',
                SHA1
               ),
        )

        def testEncrypt1(self):
            # Verify encryption using all test vectors
            for test in self._testData:
                # Build the key
                comps = [int(rws(test[0][x]), 16) for x in ('n', 'e')]
                key = RSA.construct(comps)

                # RNG that takes its random numbers from a pool given
                # at initialization
                class randGen:

                    def __init__(self, data):
                        self.data = data
                        self.idx = 0

                    def __call__(self, N):
                        r = self.data[self.idx:N]
                        self.idx += N
                        return r

                # The real test
                cipher = PKCS.new(key, test[4], randfunc=randGen(t2b(test[3])))
                ct = cipher.encrypt(t2b(test[1]))
                self.assertEqual(ct, t2b(test[2]))

        def testEncrypt2(self):
            # Verify that encryption fails if plaintext is too long
            pt = '\x00'*(128-2*20-2+1)
            cipher = PKCS.new(self.key1024)
            self.assertRaises(ValueError, cipher.encrypt, pt)

        def testDecrypt1(self):
            # Verify decryption using all test vectors
            for test in self._testData:
                # Build the key
                comps = [int(rws(test[0][x]),16) for x in ('n', 'e', 'd')]
                key = RSA.construct(comps)
                # The real test
                cipher = PKCS.new(key, test[4])
                pt = cipher.decrypt(t2b(test[2]))
                self.assertEqual(pt, t2b(test[1]))

        def testDecrypt2(self):
            # Simplest possible negative tests
            for ct_size in (127, 128, 129):
                cipher = PKCS.new(self.key1024)
                self.assertRaises(ValueError, cipher.decrypt, bchr(0x00)*ct_size)

        def testEncryptDecrypt1(self):
            # Encrypt/Decrypt messages of length [0..128-2*20-2]
            for pt_len in range(0, 128-2*20-2):
                pt = self.rng(pt_len)
                cipher = PKCS.new(self.key1024)
                ct = cipher.encrypt(pt)
                pt2 = cipher.decrypt(ct)
                self.assertEqual(pt, pt2)

        def testEncryptDecrypt2(self):
            # Helper function to monitor what's requested from RNG
            global asked

            def localRng(N):
                global asked
                asked += N
                return self.rng(N)

            # Verify that OAEP is friendly to all hashes
            for hashmod in (MD2, MD5, SHA1, SHA256, RIPEMD160):
                # Verify that encrypt() asks for as many random bytes
                # as the hash output size
                asked = 0
                pt = self.rng(40)
                cipher = PKCS.new(self.key1024, hashmod, randfunc=localRng)
                ct = cipher.encrypt(pt)
                self.assertEqual(cipher.decrypt(ct), pt)
                self.assertEqual(asked, hashmod.digest_size)

        def testEncryptDecrypt3(self):
            # Verify that OAEP supports labels
            pt = self.rng(35)
            xlabel = self.rng(22)
            cipher = PKCS.new(self.key1024, label=xlabel)
            ct = cipher.encrypt(pt)
            self.assertEqual(cipher.decrypt(ct), pt)

        def testEncryptDecrypt4(self):
            # Verify that encrypt() uses the custom MGF
            global mgfcalls
            # Helper function to monitor what's requested from MGF

            def newMGF(seed, maskLen):
                global mgfcalls
                mgfcalls += 1
                return b'\x00' * maskLen

            mgfcalls = 0
            pt = self.rng(32)
            cipher = PKCS.new(self.key1024, mgfunc=newMGF)
            ct = cipher.encrypt(pt)
            self.assertEqual(mgfcalls, 2)
            self.assertEqual(cipher.decrypt(ct), pt)

        def testByteArray(self):
            pt = b("XER")
            cipher = PKCS.new(self.key1024)
            ct = cipher.encrypt(bytearray(pt))
            pt2 = cipher.decrypt(bytearray(ct))
            self.assertEqual(pt, pt2)

        def testMemoryview(self):
            pt = b("XER")
            cipher = PKCS.new(self.key1024)
            ct = cipher.encrypt(memoryview(bytearray(pt)))
            pt2 = cipher.decrypt(memoryview(bytearray(ct)))
            self.assertEqual(pt, pt2)


class TestVectorsWycheproof(unittest.TestCase):

    def __init__(self, wycheproof_warnings, skip_slow_tests):
        unittest.TestCase.__init__(self)
        self._wycheproof_warnings = wycheproof_warnings
        self._skip_slow_tests = skip_slow_tests
        self._id = "None"

    def load_tests(self, filename):

        def filter_rsa(group):
            return RSA.import_key(group['privateKeyPem'])

        def filter_sha(group):
            if group['sha'] == "SHA-1":
                return SHA1
            elif group['sha'] == "SHA-224":
                return SHA224
            elif group['sha'] == "SHA-256":
                return SHA256
            elif group['sha'] == "SHA-384":
                return SHA384
            elif group['sha'] == "SHA-512":
                return SHA512
            else:
                raise ValueError("Unknown sha " + group['sha'])

        def filter_mgf(group):
            if group['mgfSha'] == "SHA-1":
                return lambda x, y: MGF1(x, y, SHA1)
            elif group['mgfSha'] == "SHA-224":
                return lambda x, y: MGF1(x, y, SHA224)
            elif group['mgfSha'] == "SHA-256":
                return lambda x, y: MGF1(x, y, SHA256)
            elif group['mgfSha'] == "SHA-384":
                return lambda x, y: MGF1(x, y, SHA384)
            elif group['mgfSha'] == "SHA-512":
                return lambda x, y: MGF1(x, y, SHA512)
            else:
                raise ValueError("Unknown mgf/sha " + group['mgfSha'])

        def filter_algo(group):
            return "%s with MGF1/%s" % (group['sha'], group['mgfSha'])

        result = load_test_vectors_wycheproof(("Cipher", "wycheproof"),
                                              filename,
                                              "Wycheproof PKCS#1 OAEP (%s)" % filename,
                                              group_tag={'rsa_key': filter_rsa,
                                                         'hash_mod': filter_sha,
                                                         'mgf': filter_mgf,
                                                         'algo': filter_algo}
                                              )
        return result

    def setUp(self):
        self.tv = []
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha1_mgf1sha1_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha224_mgf1sha1_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha224_mgf1sha224_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha256_mgf1sha1_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha256_mgf1sha256_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha384_mgf1sha1_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha384_mgf1sha384_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha512_mgf1sha1_test.json"))
        self.tv.extend(self.load_tests("rsa_oaep_2048_sha512_mgf1sha512_test.json"))
        if not self._skip_slow_tests:
            self.tv.extend(self.load_tests("rsa_oaep_3072_sha256_mgf1sha1_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_3072_sha256_mgf1sha256_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_3072_sha512_mgf1sha1_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_3072_sha512_mgf1sha512_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_4096_sha256_mgf1sha1_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_4096_sha256_mgf1sha256_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_4096_sha512_mgf1sha1_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_4096_sha512_mgf1sha512_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_4096_sha512_mgf1sha512_test.json"))
            self.tv.extend(self.load_tests("rsa_oaep_misc_test.json"))

    def shortDescription(self):
        return self._id

    def warn(self, tv):
        if tv.warning and self._wycheproof_warnings:
            import warnings
            warnings.warn("Wycheproof warning: %s (%s)" % (self._id, tv.comment))

    def test_decrypt(self, tv):
        self._id = "Wycheproof Decrypt %s Test #%s" % (tv.algo, tv.id)

        cipher = PKCS.new(tv.rsa_key, hashAlgo=tv.hash_mod, mgfunc=tv.mgf, label=tv.label)
        try:
            pt = cipher.decrypt(tv.ct)
        except ValueError:
            assert not tv.valid
        else:
            assert tv.valid
            self.assertEqual(pt, tv.msg)
            self.warn(tv)

    def runTest(self):

        for tv in self.tv:
            self.test_decrypt(tv)


def get_tests(config={}):
    skip_slow_tests = not config.get('slow_tests')
    wycheproof_warnings = config.get('wycheproof_warnings')

    tests = []
    tests += list_test_cases(PKCS1_OAEP_Tests)
    tests += [TestVectorsWycheproof(wycheproof_warnings, skip_slow_tests)]
    return tests


if __name__ == '__main__':
    def suite():
        unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
