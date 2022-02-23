# -*- coding: utf-8 -*-
#
#  SelfTest/PublicKey/test_importKey.py: Self-test for importing RSA keys
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

import os
import re
import errno
import warnings
import unittest

from Crypto.PublicKey import RSA
from Crypto.SelfTest.st_common import a2b_hex, list_test_cases
from Crypto.Util.py3compat import b, tostr, FileNotFoundError
from Crypto.Util.number import inverse
from Crypto.Util import asn1

try:
    import pycryptodome_test_vectors  # type: ignore
    test_vectors_available = True
except ImportError:
    test_vectors_available = False


def load_file(file_name, mode="rb"):
    results = None

    try:
        if not test_vectors_available:
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    file_name)

        dir_comps = ("PublicKey", "RSA")
        init_dir = os.path.dirname(pycryptodome_test_vectors.__file__)
        full_file_name = os.path.join(os.path.join(init_dir, *dir_comps), file_name)
        with open(full_file_name, mode) as file_in:
            results = file_in.read()

    except FileNotFoundError:
        warnings.warn("Warning: skipping extended tests for RSA",
                      UserWarning,
                      stacklevel=2)

    return results


def der2pem(der, text='PUBLIC'):
    import binascii
    chunks = [binascii.b2a_base64(der[i:i+48]) for i in range(0, len(der), 48)]
    pem = b('-----BEGIN %s KEY-----\n' % text)
    pem += b('').join(chunks)
    pem += b('-----END %s KEY-----' % text)
    return pem


class ImportKeyTests(unittest.TestCase):
    # 512-bit RSA key generated with openssl
    rsaKeyPEM = u'''-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBAL8eJ5AKoIsjURpcEoGubZMxLD7+kT+TLr7UkvEtFrRhDDKMtuII
q19FrL4pUIMymPMSLBn3hJLe30Dw48GQM4UCAwEAAQJACUSDEp8RTe32ftq8IwG8
Wojl5mAd1wFiIOrZ/Uv8b963WJOJiuQcVN29vxU5+My9GPZ7RA3hrDBEAoHUDPrI
OQIhAPIPLz4dphiD9imAkivY31Rc5AfHJiQRA7XixTcjEkojAiEAyh/pJHks/Mlr
+rdPNEpotBjfV4M4BkgGAA/ipcmaAjcCIQCHvhwwKVBLzzTscT2HeUdEeBMoiXXK
JACAr3sJQJGxIQIgarRp+m1WSKV1MciwMaTOnbU7wxFs9DP1pva76lYBzgUCIQC9
n0CnZCJ6IZYqSt0H5N7+Q+2Ro64nuwV/OSQfM6sBwQ==
-----END RSA PRIVATE KEY-----'''

    # As above, but this is actually an unencrypted PKCS#8 key
    rsaKeyPEM8 = u'''-----BEGIN PRIVATE KEY-----
MIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7AgEAAkEAvx4nkAqgiyNRGlwS
ga5tkzEsPv6RP5MuvtSS8S0WtGEMMoy24girX0WsvilQgzKY8xIsGfeEkt7fQPDj
wZAzhQIDAQABAkAJRIMSnxFN7fZ+2rwjAbxaiOXmYB3XAWIg6tn9S/xv3rdYk4mK
5BxU3b2/FTn4zL0Y9ntEDeGsMEQCgdQM+sg5AiEA8g8vPh2mGIP2KYCSK9jfVFzk
B8cmJBEDteLFNyMSSiMCIQDKH+kkeSz8yWv6t080Smi0GN9XgzgGSAYAD+KlyZoC
NwIhAIe+HDApUEvPNOxxPYd5R0R4EyiJdcokAICvewlAkbEhAiBqtGn6bVZIpXUx
yLAxpM6dtTvDEWz0M/Wm9rvqVgHOBQIhAL2fQKdkInohlipK3Qfk3v5D7ZGjrie7
BX85JB8zqwHB
-----END PRIVATE KEY-----'''

    # The same RSA private key as in rsaKeyPEM, but now encrypted
    rsaKeyEncryptedPEM = (

        # PEM encryption
        # With DES and passphrase 'test'
        ('test', u'''-----BEGIN RSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: DES-CBC,AF8F9A40BD2FA2FC

Ckl9ex1kaVEWhYC2QBmfaF+YPiR4NFkRXA7nj3dcnuFEzBnY5XULupqQpQI3qbfA
u8GYS7+b3toWWiHZivHbAAUBPDIZG9hKDyB9Sq2VMARGsX1yW1zhNvZLIiVJzUHs
C6NxQ1IJWOXzTew/xM2I26kPwHIvadq+/VaT8gLQdjdH0jOiVNaevjWnLgrn1mLP
BCNRMdcexozWtAFNNqSzfW58MJL2OdMi21ED184EFytIc1BlB+FZiGZduwKGuaKy
9bMbdb/1PSvsSzPsqW7KSSrTw6MgJAFJg6lzIYvR5F4poTVBxwBX3+EyEmShiaNY
IRX3TgQI0IjrVuLmvlZKbGWP18FXj7I7k9tSsNOOzllTTdq3ny5vgM3A+ynfAaxp
dysKznQ6P+IoqML1WxAID4aGRMWka+uArOJ148Rbj9s=
-----END RSA PRIVATE KEY-----'''),

        # PKCS8 encryption
        ('winter', u'''-----BEGIN ENCRYPTED PRIVATE KEY-----
MIIBpjBABgkqhkiG9w0BBQ0wMzAbBgkqhkiG9w0BBQwwDgQIeZIsbW3O+JcCAggA
MBQGCCqGSIb3DQMHBAgSM2p0D8FilgSCAWBhFyP2tiGKVpGj3mO8qIBzinU60ApR
3unvP+N6j7LVgnV2lFGaXbJ6a1PbQXe+2D6DUyBLo8EMXrKKVLqOMGkFMHc0UaV6
R6MmrsRDrbOqdpTuVRW+NVd5J9kQQh4xnfU/QrcPPt7vpJvSf4GzG0n666Ki50OV
M/feuVlIiyGXY6UWdVDpcOV72cq02eNUs/1JWdh2uEBvA9fCL0c07RnMrdT+CbJQ
NjJ7f8ULtp7xvR9O3Al/yJ4Wv3i4VxF1f3MCXzhlUD4I0ONlr0kJWgeQ80q/cWhw
ntvgJwnCn2XR1h6LA8Wp+0ghDTsL2NhJpWd78zClGhyU4r3hqu1XDjoXa7YCXCix
jCV15+ViDJzlNCwg+W6lRg18sSLkCT7alviIE0U5tHc6UPbbHwT5QqAxAABaP+nZ
CGqJGyiwBzrKebjgSm/KRd4C91XqcsysyH2kKPfT51MLAoD4xelOURBP
-----END ENCRYPTED PRIVATE KEY-----'''
        ),
    )

    rsaPublicKeyPEM = u'''-----BEGIN PUBLIC KEY-----
MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAL8eJ5AKoIsjURpcEoGubZMxLD7+kT+T
Lr7UkvEtFrRhDDKMtuIIq19FrL4pUIMymPMSLBn3hJLe30Dw48GQM4UCAwEAAQ==
-----END PUBLIC KEY-----'''

    # Obtained using 'ssh-keygen -i -m PKCS8 -f rsaPublicKeyPEM'
    rsaPublicKeyOpenSSH = b('''ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQC/HieQCqCLI1EaXBKBrm2TMSw+/pE/ky6+1JLxLRa0YQwyjLbiCKtfRay+KVCDMpjzEiwZ94SS3t9A8OPBkDOF comment\n''')

    # The private key, in PKCS#1 format encoded with DER
    rsaKeyDER = a2b_hex(
    '''3082013b020100024100bf1e27900aa08b23511a5c1281ae6d93312c3efe
    913f932ebed492f12d16b4610c328cb6e208ab5f45acbe2950833298f312
    2c19f78492dedf40f0e3c190338502030100010240094483129f114dedf6
    7edabc2301bc5a88e5e6601dd7016220ead9fd4bfc6fdeb75893898ae41c
    54ddbdbf1539f8ccbd18f67b440de1ac30440281d40cfac839022100f20f
    2f3e1da61883f62980922bd8df545ce407c726241103b5e2c53723124a23
    022100ca1fe924792cfcc96bfab74f344a68b418df578338064806000fe2
    a5c99a023702210087be1c3029504bcf34ec713d877947447813288975ca
    240080af7b094091b12102206ab469fa6d5648a57531c8b031a4ce9db53b
    c3116cf433f5a6f6bbea5601ce05022100bd9f40a764227a21962a4add07
    e4defe43ed91a3ae27bb057f39241f33ab01c1
    '''.replace(" ",""))

    # The private key, in unencrypted PKCS#8 format encoded with DER
    rsaKeyDER8 = a2b_hex(
    '''30820155020100300d06092a864886f70d01010105000482013f3082013
    b020100024100bf1e27900aa08b23511a5c1281ae6d93312c3efe913f932
    ebed492f12d16b4610c328cb6e208ab5f45acbe2950833298f3122c19f78
    492dedf40f0e3c190338502030100010240094483129f114dedf67edabc2
    301bc5a88e5e6601dd7016220ead9fd4bfc6fdeb75893898ae41c54ddbdb
    f1539f8ccbd18f67b440de1ac30440281d40cfac839022100f20f2f3e1da
    61883f62980922bd8df545ce407c726241103b5e2c53723124a23022100c
    a1fe924792cfcc96bfab74f344a68b418df578338064806000fe2a5c99a0
    23702210087be1c3029504bcf34ec713d877947447813288975ca240080a
    f7b094091b12102206ab469fa6d5648a57531c8b031a4ce9db53bc3116cf
    433f5a6f6bbea5601ce05022100bd9f40a764227a21962a4add07e4defe4
    3ed91a3ae27bb057f39241f33ab01c1
    '''.replace(" ",""))

    rsaPublicKeyDER = a2b_hex(
    '''305c300d06092a864886f70d0101010500034b003048024100bf1e27900a
    a08b23511a5c1281ae6d93312c3efe913f932ebed492f12d16b4610c328c
    b6e208ab5f45acbe2950833298f3122c19f78492dedf40f0e3c190338502
    03010001
    '''.replace(" ",""))

    n = int('BF 1E 27 90 0A A0 8B 23 51 1A 5C 12 81 AE 6D 93 31 2C 3E FE 91 3F 93 2E BE D4 92 F1 2D 16 B4 61 0C 32 8C B6 E2 08 AB 5F 45 AC BE 29 50 83 32 98 F3 12 2C 19 F7 84 92 DE DF 40 F0 E3 C1 90 33 85'.replace(" ",""),16)
    e = 65537
    d = int('09 44 83 12 9F 11 4D ED F6 7E DA BC 23 01 BC 5A 88 E5 E6 60 1D D7 01 62 20 EA D9 FD 4B FC 6F DE B7 58 93 89 8A E4 1C 54 DD BD BF 15 39 F8 CC BD 18 F6 7B 44 0D E1 AC 30 44 02 81 D4 0C FA C8 39'.replace(" ",""),16)
    p = int('00 F2 0F 2F 3E 1D A6 18 83 F6 29 80 92 2B D8 DF 54 5C E4 07 C7 26 24 11 03 B5 E2 C5 37 23 12 4A 23'.replace(" ",""),16)
    q = int('00 CA 1F E9 24 79 2C FC C9 6B FA B7 4F 34 4A 68 B4 18 DF 57 83 38 06 48 06 00 0F E2 A5 C9 9A 02 37'.replace(" ",""),16)

    # This is q^{-1} mod p). fastmath and slowmath use pInv (p^{-1}
    # mod q) instead!
    qInv = int('00 BD 9F 40 A7 64 22 7A 21 96 2A 4A DD 07 E4 DE FE 43 ED 91 A3 AE 27 BB 05 7F 39 24 1F 33 AB 01 C1'.replace(" ",""),16)
    pInv = inverse(p,q)

    def testImportKey1(self):
        """Verify import of RSAPrivateKey DER SEQUENCE"""
        key = RSA.importKey(self.rsaKeyDER)
        self.assertTrue(key.has_private())
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)
        self.assertEqual(key.d, self.d)
        self.assertEqual(key.p, self.p)
        self.assertEqual(key.q, self.q)

    def testImportKey2(self):
        """Verify import of SubjectPublicKeyInfo DER SEQUENCE"""
        key = RSA.importKey(self.rsaPublicKeyDER)
        self.assertFalse(key.has_private())
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)

    def testImportKey3unicode(self):
        """Verify import of RSAPrivateKey DER SEQUENCE, encoded with PEM as unicode"""
        key = RSA.importKey(self.rsaKeyPEM)
        self.assertEqual(key.has_private(),True) # assert_
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)
        self.assertEqual(key.d, self.d)
        self.assertEqual(key.p, self.p)
        self.assertEqual(key.q, self.q)

    def testImportKey3bytes(self):
        """Verify import of RSAPrivateKey DER SEQUENCE, encoded with PEM as byte string"""
        key = RSA.importKey(b(self.rsaKeyPEM))
        self.assertEqual(key.has_private(),True) # assert_
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)
        self.assertEqual(key.d, self.d)
        self.assertEqual(key.p, self.p)
        self.assertEqual(key.q, self.q)

    def testImportKey4unicode(self):
        """Verify import of RSAPrivateKey DER SEQUENCE, encoded with PEM as unicode"""
        key = RSA.importKey(self.rsaPublicKeyPEM)
        self.assertEqual(key.has_private(),False) # assertFalse
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)

    def testImportKey4bytes(self):
        """Verify import of SubjectPublicKeyInfo DER SEQUENCE, encoded with PEM as byte string"""
        key = RSA.importKey(b(self.rsaPublicKeyPEM))
        self.assertEqual(key.has_private(),False) # assertFalse
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)

    def testImportKey5(self):
        """Verifies that the imported key is still a valid RSA pair"""
        key = RSA.importKey(self.rsaKeyPEM)
        idem = key._encrypt(key._decrypt(89))
        self.assertEqual(idem, 89)

    def testImportKey6(self):
        """Verifies that the imported key is still a valid RSA pair"""
        key = RSA.importKey(self.rsaKeyDER)
        idem = key._encrypt(key._decrypt(65))
        self.assertEqual(idem, 65)

    def testImportKey7(self):
        """Verify import of OpenSSH public key"""
        key = RSA.importKey(self.rsaPublicKeyOpenSSH)
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)

    def testImportKey8(self):
        """Verify import of encrypted PrivateKeyInfo DER SEQUENCE"""
        for t in self.rsaKeyEncryptedPEM:
            key = RSA.importKey(t[1], t[0])
            self.assertTrue(key.has_private())
            self.assertEqual(key.n, self.n)
            self.assertEqual(key.e, self.e)
            self.assertEqual(key.d, self.d)
            self.assertEqual(key.p, self.p)
            self.assertEqual(key.q, self.q)

    def testImportKey9(self):
        """Verify import of unencrypted PrivateKeyInfo DER SEQUENCE"""
        key = RSA.importKey(self.rsaKeyDER8)
        self.assertTrue(key.has_private())
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)
        self.assertEqual(key.d, self.d)
        self.assertEqual(key.p, self.p)
        self.assertEqual(key.q, self.q)

    def testImportKey10(self):
        """Verify import of unencrypted PrivateKeyInfo DER SEQUENCE, encoded with PEM"""
        key = RSA.importKey(self.rsaKeyPEM8)
        self.assertTrue(key.has_private())
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)
        self.assertEqual(key.d, self.d)
        self.assertEqual(key.p, self.p)
        self.assertEqual(key.q, self.q)

    def testImportKey11(self):
        """Verify import of RSAPublicKey DER SEQUENCE"""
        der = asn1.DerSequence([17, 3]).encode()
        key = RSA.importKey(der)
        self.assertEqual(key.n, 17)
        self.assertEqual(key.e, 3)

    def testImportKey12(self):
        """Verify import of RSAPublicKey DER SEQUENCE, encoded with PEM"""
        der = asn1.DerSequence([17, 3]).encode()
        pem = der2pem(der)
        key = RSA.importKey(pem)
        self.assertEqual(key.n, 17)
        self.assertEqual(key.e, 3)

    def test_import_key_windows_cr_lf(self):
        pem_cr_lf = "\r\n".join(self.rsaKeyPEM.splitlines())
        key = RSA.importKey(pem_cr_lf)
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)
        self.assertEqual(key.d, self.d)
        self.assertEqual(key.p, self.p)
        self.assertEqual(key.q, self.q)

    def test_import_empty(self):
        self.assertRaises(ValueError, RSA.import_key, b"")

    ###
    def testExportKey1(self):
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        derKey = key.export_key("DER")
        self.assertEqual(derKey, self.rsaKeyDER)

    def testExportKey2(self):
        key = RSA.construct([self.n, self.e])
        derKey = key.export_key("DER")
        self.assertEqual(derKey, self.rsaPublicKeyDER)

    def testExportKey3(self):
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        pemKey = key.export_key("PEM")
        self.assertEqual(pemKey, b(self.rsaKeyPEM))

    def testExportKey4(self):
        key = RSA.construct([self.n, self.e])
        pemKey = key.export_key("PEM")
        self.assertEqual(pemKey, b(self.rsaPublicKeyPEM))

    def testExportKey5(self):
        key = RSA.construct([self.n, self.e])
        openssh_1 = key.export_key("OpenSSH").split()
        openssh_2 = self.rsaPublicKeyOpenSSH.split()
        self.assertEqual(openssh_1[0], openssh_2[0])
        self.assertEqual(openssh_1[1], openssh_2[1])

    def testExportKey7(self):
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        derKey = key.export_key("DER", pkcs=8)
        self.assertEqual(derKey, self.rsaKeyDER8)

    def testExportKey8(self):
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        pemKey = key.export_key("PEM", pkcs=8)
        self.assertEqual(pemKey, b(self.rsaKeyPEM8))

    def testExportKey9(self):
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        self.assertRaises(ValueError, key.export_key, "invalid-format")

    def testExportKey10(self):
        # Export and re-import the encrypted key. It must match.
        # PEM envelope, PKCS#1, old PEM encryption
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        outkey = key.export_key('PEM', 'test')
        self.assertTrue(tostr(outkey).find('4,ENCRYPTED')!=-1)
        self.assertTrue(tostr(outkey).find('BEGIN RSA PRIVATE KEY')!=-1)
        inkey = RSA.importKey(outkey, 'test')
        self.assertEqual(key.n, inkey.n)
        self.assertEqual(key.e, inkey.e)
        self.assertEqual(key.d, inkey.d)

    def testExportKey11(self):
        # Export and re-import the encrypted key. It must match.
        # PEM envelope, PKCS#1, old PEM encryption
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        outkey = key.export_key('PEM', 'test', pkcs=1)
        self.assertTrue(tostr(outkey).find('4,ENCRYPTED')!=-1)
        self.assertTrue(tostr(outkey).find('BEGIN RSA PRIVATE KEY')!=-1)
        inkey = RSA.importKey(outkey, 'test')
        self.assertEqual(key.n, inkey.n)
        self.assertEqual(key.e, inkey.e)
        self.assertEqual(key.d, inkey.d)

    def testExportKey12(self):
        # Export and re-import the encrypted key. It must match.
        # PEM envelope, PKCS#8, old PEM encryption
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        outkey = key.export_key('PEM', 'test', pkcs=8)
        self.assertTrue(tostr(outkey).find('4,ENCRYPTED')!=-1)
        self.assertTrue(tostr(outkey).find('BEGIN PRIVATE KEY')!=-1)
        inkey = RSA.importKey(outkey, 'test')
        self.assertEqual(key.n, inkey.n)
        self.assertEqual(key.e, inkey.e)
        self.assertEqual(key.d, inkey.d)

    def testExportKey13(self):
        # Export and re-import the encrypted key. It must match.
        # PEM envelope, PKCS#8, PKCS#8 encryption
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        outkey = key.export_key('PEM', 'test', pkcs=8,
                protection='PBKDF2WithHMAC-SHA1AndDES-EDE3-CBC')
        self.assertTrue(tostr(outkey).find('4,ENCRYPTED')==-1)
        self.assertTrue(tostr(outkey).find('BEGIN ENCRYPTED PRIVATE KEY')!=-1)
        inkey = RSA.importKey(outkey, 'test')
        self.assertEqual(key.n, inkey.n)
        self.assertEqual(key.e, inkey.e)
        self.assertEqual(key.d, inkey.d)

    def testExportKey14(self):
        # Export and re-import the encrypted key. It must match.
        # DER envelope, PKCS#8, PKCS#8 encryption
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        outkey = key.export_key('DER', 'test', pkcs=8)
        inkey = RSA.importKey(outkey, 'test')
        self.assertEqual(key.n, inkey.n)
        self.assertEqual(key.e, inkey.e)
        self.assertEqual(key.d, inkey.d)

    def testExportKey15(self):
        # Verify that that error an condition is detected when trying to
        # use a password with DER encoding and PKCS#1.
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        self.assertRaises(ValueError, key.export_key, 'DER', 'test', 1)

    def test_import_key(self):
        """Verify that import_key is an alias to importKey"""
        key = RSA.import_key(self.rsaPublicKeyDER)
        self.assertFalse(key.has_private())
        self.assertEqual(key.n, self.n)
        self.assertEqual(key.e, self.e)

    def test_import_key_ba_mv(self):
        """Verify that import_key can be used on bytearrays and memoryviews"""
        key = RSA.import_key(bytearray(self.rsaPublicKeyDER))
        key = RSA.import_key(memoryview(self.rsaPublicKeyDER))

    def test_exportKey(self):
        key = RSA.construct([self.n, self.e, self.d, self.p, self.q, self.pInv])
        self.assertEqual(key.export_key(), key.exportKey())


class ImportKeyFromX509Cert(unittest.TestCase):

    def test_x509v1(self):

        # Sample V1 certificate with a 1024 bit RSA key
        x509_v1_cert = """
-----BEGIN CERTIFICATE-----
MIICOjCCAaMCAQEwDQYJKoZIhvcNAQEEBQAwfjENMAsGA1UEChMEQWNtZTELMAkG
A1UECxMCUkQxHDAaBgkqhkiG9w0BCQEWDXNwYW1AYWNtZS5vcmcxEzARBgNVBAcT
Ck1ldHJvcG9saXMxETAPBgNVBAgTCE5ldyBZb3JrMQswCQYDVQQGEwJVUzENMAsG
A1UEAxMEdGVzdDAeFw0xNDA3MTExOTU3MjRaFw0xNzA0MDYxOTU3MjRaME0xCzAJ
BgNVBAYTAlVTMREwDwYDVQQIEwhOZXcgWW9yazENMAsGA1UEChMEQWNtZTELMAkG
A1UECxMCUkQxDzANBgNVBAMTBmxhdHZpYTCBnzANBgkqhkiG9w0BAQEFAAOBjQAw
gYkCgYEAyG+kytdRj3TFbRmHDYp3TXugVQ81chew0qeOxZWOz80IjtWpgdOaCvKW
NCuc8wUR9BWrEQW+39SaRMLiQfQtyFSQZijc3nsEBu/Lo4uWZ0W/FHDRVSvkJA/V
Ex5NL5ikI+wbUeCV5KajGNDalZ8F1pk32+CBs8h1xNx5DyxuEHUCAwEAATANBgkq
hkiG9w0BAQQFAAOBgQCVQF9Y//Q4Psy+umEM38pIlbZ2hxC5xNz/MbVPwuCkNcGn
KYNpQJP+JyVTsPpO8RLZsAQDzRueMI3S7fbbwTzAflN0z19wvblvu93xkaBytVok
9VBAH28olVhy9b1MMeg2WOt5sUEQaFNPnwwsyiY9+HsRpvpRnPSQF+kyYVsshQ==
-----END CERTIFICATE-----
        """.strip()

        # RSA public key as dumped by openssl
        exponent = 65537
        modulus_str = """
00:c8:6f:a4:ca:d7:51:8f:74:c5:6d:19:87:0d:8a:
77:4d:7b:a0:55:0f:35:72:17:b0:d2:a7:8e:c5:95:
8e:cf:cd:08:8e:d5:a9:81:d3:9a:0a:f2:96:34:2b:
9c:f3:05:11:f4:15:ab:11:05:be:df:d4:9a:44:c2:
e2:41:f4:2d:c8:54:90:66:28:dc:de:7b:04:06:ef:
cb:a3:8b:96:67:45:bf:14:70:d1:55:2b:e4:24:0f:
d5:13:1e:4d:2f:98:a4:23:ec:1b:51:e0:95:e4:a6:
a3:18:d0:da:95:9f:05:d6:99:37:db:e0:81:b3:c8:
75:c4:dc:79:0f:2c:6e:10:75
        """
        modulus = int(re.sub("[^0-9a-f]","", modulus_str), 16)

        key = RSA.importKey(x509_v1_cert)
        self.assertEqual(key.e, exponent)
        self.assertEqual(key.n, modulus)
        self.assertFalse(key.has_private())

    def test_x509v3(self):

        # Sample V3 certificate with a 1024 bit RSA key
        x509_v3_cert = """
-----BEGIN CERTIFICATE-----
MIIEcjCCAlqgAwIBAgIBATANBgkqhkiG9w0BAQsFADBhMQswCQYDVQQGEwJVUzEL
MAkGA1UECAwCTUQxEjAQBgNVBAcMCUJhbHRpbW9yZTEQMA4GA1UEAwwHVGVzdCBD
QTEfMB0GCSqGSIb3DQEJARYQdGVzdEBleGFtcGxlLmNvbTAeFw0xNDA3MTIwOTM1
MTJaFw0xNzA0MDcwOTM1MTJaMEQxCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJNRDES
MBAGA1UEBwwJQmFsdGltb3JlMRQwEgYDVQQDDAtUZXN0IFNlcnZlcjCBnzANBgkq
hkiG9w0BAQEFAAOBjQAwgYkCgYEA/S7GJV2OcFdyNMQ4K75KrYFtMEn3VnEFdPHa
jyS37XlMxSh0oS4GeTGVUCJInl5Cpsv8WQdh03FfeOdvzp5IZ46OcjeOPiWnmjgl
2G5j7e2bDH7RSchGV+OD6Fb1Agvuu2/9iy8fdf3rPQ/7eAddzKUrzwacVbnW+tg2
QtSXKRcCAwEAAaOB1TCB0jAdBgNVHQ4EFgQU/WwCX7FfWMIPDFfJ+I8a2COG+l8w
HwYDVR0jBBgwFoAUa0hkif3RMaraiWtsOOZZlLu9wJwwCQYDVR0TBAIwADALBgNV
HQ8EBAMCBeAwSgYDVR0RBEMwQYILZXhhbXBsZS5jb22CD3d3dy5leGFtcGxlLmNv
bYIQbWFpbC5leGFtcGxlLmNvbYIPZnRwLmV4YW1wbGUuY29tMCwGCWCGSAGG+EIB
DQQfFh1PcGVuU1NMIEdlbmVyYXRlZCBDZXJ0aWZpY2F0ZTANBgkqhkiG9w0BAQsF
AAOCAgEAvO6xfdsGbnoK4My3eJthodTAjMjPwFVY133LH04QLcCv54TxKhtUg1fi
PgdjVe1HpTytPBfXy2bSZbXAN0abZCtw1rYrnn7o1g2pN8iypVq3zVn0iMTzQzxs
zEPO3bpR/UhNSf90PmCsS5rqZpAAnXSaAy1ClwHWk/0eG2pYkhE1m1ABVMN2lsAW
e9WxGk6IFqaI9O37NYQwmEypMs4DC+ECJEvbPFiqi3n0gbXCZJJ6omDA5xJldaYK
Oa7KR3s/qjBsu9UAiWpLBuFoSTHIF2aeRKRFmUdmzwo43eVPep65pY6eQ4AdL2RF
rqEuINbGlzI5oQyYhu71IwB+iPZXaZZPlwjLgOsuad/p2hOgDb5WxUi8FnDPursQ
ujfpIpmrOP/zpvvQWnwePI3lI+5n41kTBSbefXEdv6rXpHk3QRzB90uPxnXPdxSC
16ASA8bQT5an/1AgoE3k9CrcD2K0EmgaX0YI0HUhkyzbkg34EhpWJ6vvRUbRiNRo
9cIbt/ya9Y9u0Ja8GLXv6dwX0l0IdJMkL8KifXUFAVCujp1FBrr/gdmwQn8itANy
+qbnWSxmOvtaY0zcaFAcONuHva0h51/WqXOMO1eb8PhR4HIIYU8p1oBwQp7dSni8
THDi1F+GG5PsymMDj5cWK42f+QzjVw5PrVmFqqrrEoMlx8DWh5Y=
-----END CERTIFICATE-----
""".strip()

        # RSA public key as dumped by openssl
        exponent = 65537
        modulus_str = """
00:fd:2e:c6:25:5d:8e:70:57:72:34:c4:38:2b:be:
4a:ad:81:6d:30:49:f7:56:71:05:74:f1:da:8f:24:
b7:ed:79:4c:c5:28:74:a1:2e:06:79:31:95:50:22:
48:9e:5e:42:a6:cb:fc:59:07:61:d3:71:5f:78:e7:
6f:ce:9e:48:67:8e:8e:72:37:8e:3e:25:a7:9a:38:
25:d8:6e:63:ed:ed:9b:0c:7e:d1:49:c8:46:57:e3:
83:e8:56:f5:02:0b:ee:bb:6f:fd:8b:2f:1f:75:fd:
eb:3d:0f:fb:78:07:5d:cc:a5:2b:cf:06:9c:55:b9:
d6:fa:d8:36:42:d4:97:29:17
        """
        modulus = int(re.sub("[^0-9a-f]","", modulus_str), 16)

        key = RSA.importKey(x509_v3_cert)
        self.assertEqual(key.e, exponent)
        self.assertEqual(key.n, modulus)
        self.assertFalse(key.has_private())


class TestImport_2048(unittest.TestCase):

    def test_import_openssh_public(self):
        key_file_ref = load_file("rsa2048_private.pem")
        key_file = load_file("rsa2048_public_openssh.txt")

        # Skip test if test vectors are not installed
        if None in (key_file_ref, key_file):
            return

        key_ref = RSA.import_key(key_file_ref).public_key()
        key = RSA.import_key(key_file)
        self.assertEqual(key_ref, key)

    def test_import_openssh_private_clear(self):
        key_file = load_file("rsa2048_private_openssh.pem")
        key_file_old = load_file("rsa2048_private_openssh_old.pem")

        # Skip test if test vectors are not installed
        if None in (key_file_old, key_file):
            return

        key = RSA.import_key(key_file)
        key_old = RSA.import_key(key_file_old)

        self.assertEqual(key, key_old)

    def test_import_openssh_private_password(self):
        key_file = load_file("rsa2048_private_openssh_pwd.pem")
        key_file_old = load_file("rsa2048_private_openssh_pwd_old.pem")

        # Skip test if test vectors are not installed
        if None in (key_file_old, key_file):
            return

        key = RSA.import_key(key_file, b"password")
        key_old = RSA.import_key(key_file_old)
        self.assertEqual(key, key_old)


if __name__ == '__main__':
    unittest.main()


def get_tests(config={}):
    tests = []
    tests += list_test_cases(ImportKeyTests)
    tests += list_test_cases(ImportKeyFromX509Cert)
    tests += list_test_cases(TestImport_2048)
    return tests


if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
