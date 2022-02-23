# -*- coding: utf-8 -*-
#
#  SelfTest/PublicKey/test_import_DSA.py: Self-test for importing DSA keys
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
import re

from Crypto.PublicKey import DSA
from Crypto.SelfTest.st_common import *
from Crypto.Util.py3compat import *

from binascii import unhexlify

class ImportKeyTests(unittest.TestCase):

    y = 92137165128186062214622779787483327510946462589285775188003362705875131352591574106484271700740858696583623951844732128165434284507709057439633739849986759064015013893156866539696757799934634945787496920169462601722830899660681779448742875054459716726855443681559131362852474817534616736104831095601710736729
    p = 162452170958135306109773853318304545923250830605675936228618290525164105310663722368377131295055868997377338797580997938253236213714988311430600065853662861806894003694743806769284131194035848116051021923956699231855223389086646903420682639786976554552864568460372266462812137447840653688476258666833303658691
    q = 988791743931120302950649732173330531512663554851
    g = 85583152299197514738065570254868711517748965097380456700369348466136657764813442044039878840094809620913085570225318356734366886985903212775602770761953571967834823306046501307810937486758039063386311593890777319935391363872375452381836756832784184928202587843258855704771836753434368484556809100537243908232
    x = 540873410045082450874416847965843801027716145253

    def setUp(self):

        # It is easier to write test vectors in text form,
        # and convert them to byte strigs dynamically here
        for mname, mvalue in ImportKeyTests.__dict__.items():
            if mname[:4] in ('der_', 'pem_', 'ssh_'):
                if mname[:4] == 'der_':
                    mvalue = unhexlify(tobytes(mvalue))
                mvalue = tobytes(mvalue)
                setattr(self, mname, mvalue)

    # 1. SubjectPublicKeyInfo
    der_public=\
    '308201b73082012b06072a8648ce3804013082011e02818100e756ee1717f4b6'+\
    '794c7c214724a19763742c45572b4b3f8ff3b44f3be9f44ce039a2757695ec91'+\
    '5697da74ef914fcd1b05660e2419c761d639f45d2d79b802dbd23e7ab8b81b47'+\
    '9a380e1f30932584ba2a0b955032342ebc83cb5ca906e7b0d7cd6fe656cecb4c'+\
    '8b5a77123a8c6750a481e3b06057aff6aa6eba620b832d60c3021500ad32f48c'+\
    'd3ae0c45a198a61fa4b5e20320763b2302818079dfdc3d614fe635fceb7eaeae'+\
    '3718dc2efefb45282993ac6749dc83c223d8c1887296316b3b0b54466cf444f3'+\
    '4b82e3554d0b90a778faaf1306f025dae6a3e36c7f93dd5bac4052b92370040a'+\
    'ca70b8d5820599711900efbc961812c355dd9beffe0981da85c5548074b41c56'+\
    'ae43fd300d89262e4efd89943f99a651b03888038185000281810083352a69a1'+\
    '32f34843d2a0eb995bff4e2f083a73f0049d2c91ea2f0ce43d144abda48199e4'+\
    'b003c570a8af83303d45105f606c5c48d925a40ed9c2630c2fa4cdbf838539de'+\
    'b9a29f919085f2046369f627ca84b2cb1e2c7940564b670f963ab1164d4e2ca2'+\
    'bf6ffd39f12f548928bf4d2d1b5e6980b4f1be4c92a91986fba559'

    def testImportKey1(self):
        key_obj = DSA.importKey(self.der_public)
        self.assertFalse(key_obj.has_private())
        self.assertEqual(self.y, key_obj.y)
        self.assertEqual(self.p, key_obj.p)
        self.assertEqual(self.q, key_obj.q)
        self.assertEqual(self.g, key_obj.g)

    def testExportKey1(self):
        tup = (self.y, self.g, self.p, self.q)
        key = DSA.construct(tup)
        encoded = key.export_key('DER')
        self.assertEqual(self.der_public, encoded)

    # 2.
    pem_public="""\
-----BEGIN PUBLIC KEY-----
MIIBtzCCASsGByqGSM44BAEwggEeAoGBAOdW7hcX9LZ5THwhRyShl2N0LEVXK0s/
j/O0Tzvp9EzgOaJ1dpXskVaX2nTvkU/NGwVmDiQZx2HWOfRdLXm4AtvSPnq4uBtH
mjgOHzCTJYS6KguVUDI0LryDy1ypBuew181v5lbOy0yLWncSOoxnUKSB47BgV6/2
qm66YguDLWDDAhUArTL0jNOuDEWhmKYfpLXiAyB2OyMCgYB539w9YU/mNfzrfq6u
NxjcLv77RSgpk6xnSdyDwiPYwYhyljFrOwtURmz0RPNLguNVTQuQp3j6rxMG8CXa
5qPjbH+T3VusQFK5I3AECspwuNWCBZlxGQDvvJYYEsNV3Zvv/gmB2oXFVIB0tBxW
rkP9MA2JJi5O/YmUP5mmUbA4iAOBhQACgYEAgzUqaaEy80hD0qDrmVv/Ti8IOnPw
BJ0skeovDOQ9FEq9pIGZ5LADxXCor4MwPUUQX2BsXEjZJaQO2cJjDC+kzb+DhTne
uaKfkZCF8gRjafYnyoSyyx4seUBWS2cPljqxFk1OLKK/b/058S9UiSi/TS0bXmmA
tPG+TJKpGYb7pVk=
-----END PUBLIC KEY-----"""

    def testImportKey2(self):
        for pem in (self.pem_public, tostr(self.pem_public)):
            key_obj = DSA.importKey(pem)
            self.assertFalse(key_obj.has_private())
            self.assertEqual(self.y, key_obj.y)
            self.assertEqual(self.p, key_obj.p)
            self.assertEqual(self.q, key_obj.q)
            self.assertEqual(self.g, key_obj.g)

    def testExportKey2(self):
        tup = (self.y, self.g, self.p, self.q)
        key = DSA.construct(tup)
        encoded = key.export_key('PEM')
        self.assertEqual(self.pem_public, encoded)

    # 3. OpenSSL/OpenSSH format
    der_private=\
    '308201bb02010002818100e756ee1717f4b6794c7c214724a19763742c45572b'+\
    '4b3f8ff3b44f3be9f44ce039a2757695ec915697da74ef914fcd1b05660e2419'+\
    'c761d639f45d2d79b802dbd23e7ab8b81b479a380e1f30932584ba2a0b955032'+\
    '342ebc83cb5ca906e7b0d7cd6fe656cecb4c8b5a77123a8c6750a481e3b06057'+\
    'aff6aa6eba620b832d60c3021500ad32f48cd3ae0c45a198a61fa4b5e2032076'+\
    '3b2302818079dfdc3d614fe635fceb7eaeae3718dc2efefb45282993ac6749dc'+\
    '83c223d8c1887296316b3b0b54466cf444f34b82e3554d0b90a778faaf1306f0'+\
    '25dae6a3e36c7f93dd5bac4052b92370040aca70b8d5820599711900efbc9618'+\
    '12c355dd9beffe0981da85c5548074b41c56ae43fd300d89262e4efd89943f99'+\
    'a651b038880281810083352a69a132f34843d2a0eb995bff4e2f083a73f0049d'+\
    '2c91ea2f0ce43d144abda48199e4b003c570a8af83303d45105f606c5c48d925'+\
    'a40ed9c2630c2fa4cdbf838539deb9a29f919085f2046369f627ca84b2cb1e2c'+\
    '7940564b670f963ab1164d4e2ca2bf6ffd39f12f548928bf4d2d1b5e6980b4f1'+\
    'be4c92a91986fba55902145ebd9a3f0b82069d98420986b314215025756065'

    def testImportKey3(self):
        key_obj = DSA.importKey(self.der_private)
        self.assertTrue(key_obj.has_private())
        self.assertEqual(self.y, key_obj.y)
        self.assertEqual(self.p, key_obj.p)
        self.assertEqual(self.q, key_obj.q)
        self.assertEqual(self.g, key_obj.g)
        self.assertEqual(self.x, key_obj.x)

    def testExportKey3(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        encoded = key.export_key('DER', pkcs8=False)
        self.assertEqual(self.der_private, encoded)

    # 4.
    pem_private="""\
-----BEGIN DSA PRIVATE KEY-----
MIIBuwIBAAKBgQDnVu4XF/S2eUx8IUckoZdjdCxFVytLP4/ztE876fRM4DmidXaV
7JFWl9p075FPzRsFZg4kGcdh1jn0XS15uALb0j56uLgbR5o4Dh8wkyWEuioLlVAy
NC68g8tcqQbnsNfNb+ZWzstMi1p3EjqMZ1CkgeOwYFev9qpuumILgy1gwwIVAK0y
9IzTrgxFoZimH6S14gMgdjsjAoGAed/cPWFP5jX8636urjcY3C7++0UoKZOsZ0nc
g8Ij2MGIcpYxazsLVEZs9ETzS4LjVU0LkKd4+q8TBvAl2uaj42x/k91brEBSuSNw
BArKcLjVggWZcRkA77yWGBLDVd2b7/4JgdqFxVSAdLQcVq5D/TANiSYuTv2JlD+Z
plGwOIgCgYEAgzUqaaEy80hD0qDrmVv/Ti8IOnPwBJ0skeovDOQ9FEq9pIGZ5LAD
xXCor4MwPUUQX2BsXEjZJaQO2cJjDC+kzb+DhTneuaKfkZCF8gRjafYnyoSyyx4s
eUBWS2cPljqxFk1OLKK/b/058S9UiSi/TS0bXmmAtPG+TJKpGYb7pVkCFF69mj8L
ggadmEIJhrMUIVAldWBl
-----END DSA PRIVATE KEY-----"""

    def testImportKey4(self):
        for pem in (self.pem_private, tostr(self.pem_private)):
            key_obj = DSA.importKey(pem)
            self.assertTrue(key_obj.has_private())
            self.assertEqual(self.y, key_obj.y)
            self.assertEqual(self.p, key_obj.p)
            self.assertEqual(self.q, key_obj.q)
            self.assertEqual(self.g, key_obj.g)
            self.assertEqual(self.x, key_obj.x)

    def testExportKey4(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        encoded = key.export_key('PEM', pkcs8=False)
        self.assertEqual(self.pem_private, encoded)

    # 5. PKCS8 (unencrypted)
    der_pkcs8=\
    '3082014a0201003082012b06072a8648ce3804013082011e02818100e756ee17'+\
    '17f4b6794c7c214724a19763742c45572b4b3f8ff3b44f3be9f44ce039a27576'+\
    '95ec915697da74ef914fcd1b05660e2419c761d639f45d2d79b802dbd23e7ab8'+\
    'b81b479a380e1f30932584ba2a0b955032342ebc83cb5ca906e7b0d7cd6fe656'+\
    'cecb4c8b5a77123a8c6750a481e3b06057aff6aa6eba620b832d60c3021500ad'+\
    '32f48cd3ae0c45a198a61fa4b5e20320763b2302818079dfdc3d614fe635fceb'+\
    '7eaeae3718dc2efefb45282993ac6749dc83c223d8c1887296316b3b0b54466c'+\
    'f444f34b82e3554d0b90a778faaf1306f025dae6a3e36c7f93dd5bac4052b923'+\
    '70040aca70b8d5820599711900efbc961812c355dd9beffe0981da85c5548074'+\
    'b41c56ae43fd300d89262e4efd89943f99a651b03888041602145ebd9a3f0b82'+\
    '069d98420986b314215025756065'

    def testImportKey5(self):
        key_obj = DSA.importKey(self.der_pkcs8)
        self.assertTrue(key_obj.has_private())
        self.assertEqual(self.y, key_obj.y)
        self.assertEqual(self.p, key_obj.p)
        self.assertEqual(self.q, key_obj.q)
        self.assertEqual(self.g, key_obj.g)
        self.assertEqual(self.x, key_obj.x)

    def testExportKey5(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        encoded = key.export_key('DER')
        self.assertEqual(self.der_pkcs8, encoded)
        encoded = key.export_key('DER', pkcs8=True)
        self.assertEqual(self.der_pkcs8, encoded)

    # 6.
    pem_pkcs8="""\
-----BEGIN PRIVATE KEY-----
MIIBSgIBADCCASsGByqGSM44BAEwggEeAoGBAOdW7hcX9LZ5THwhRyShl2N0LEVX
K0s/j/O0Tzvp9EzgOaJ1dpXskVaX2nTvkU/NGwVmDiQZx2HWOfRdLXm4AtvSPnq4
uBtHmjgOHzCTJYS6KguVUDI0LryDy1ypBuew181v5lbOy0yLWncSOoxnUKSB47Bg
V6/2qm66YguDLWDDAhUArTL0jNOuDEWhmKYfpLXiAyB2OyMCgYB539w9YU/mNfzr
fq6uNxjcLv77RSgpk6xnSdyDwiPYwYhyljFrOwtURmz0RPNLguNVTQuQp3j6rxMG
8CXa5qPjbH+T3VusQFK5I3AECspwuNWCBZlxGQDvvJYYEsNV3Zvv/gmB2oXFVIB0
tBxWrkP9MA2JJi5O/YmUP5mmUbA4iAQWAhRevZo/C4IGnZhCCYazFCFQJXVgZQ==
-----END PRIVATE KEY-----"""

    def testImportKey6(self):
        for pem in (self.pem_pkcs8, tostr(self.pem_pkcs8)):
            key_obj = DSA.importKey(pem)
            self.assertTrue(key_obj.has_private())
            self.assertEqual(self.y, key_obj.y)
            self.assertEqual(self.p, key_obj.p)
            self.assertEqual(self.q, key_obj.q)
            self.assertEqual(self.g, key_obj.g)
            self.assertEqual(self.x, key_obj.x)

    def testExportKey6(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        encoded = key.export_key('PEM')
        self.assertEqual(self.pem_pkcs8, encoded)
        encoded = key.export_key('PEM', pkcs8=True)
        self.assertEqual(self.pem_pkcs8, encoded)

    # 7. OpenSSH/RFC4253
    ssh_pub="""ssh-dss AAAAB3NzaC1kc3MAAACBAOdW7hcX9LZ5THwhRyShl2N0LEVXK0s/j/O0Tzvp9EzgOaJ1dpXskVaX2nTvkU/NGwVmDiQZx2HWOfRdLXm4AtvSPnq4uBtHmjgOHzCTJYS6KguVUDI0LryDy1ypBuew181v5lbOy0yLWncSOoxnUKSB47BgV6/2qm66YguDLWDDAAAAFQCtMvSM064MRaGYph+kteIDIHY7IwAAAIB539w9YU/mNfzrfq6uNxjcLv77RSgpk6xnSdyDwiPYwYhyljFrOwtURmz0RPNLguNVTQuQp3j6rxMG8CXa5qPjbH+T3VusQFK5I3AECspwuNWCBZlxGQDvvJYYEsNV3Zvv/gmB2oXFVIB0tBxWrkP9MA2JJi5O/YmUP5mmUbA4iAAAAIEAgzUqaaEy80hD0qDrmVv/Ti8IOnPwBJ0skeovDOQ9FEq9pIGZ5LADxXCor4MwPUUQX2BsXEjZJaQO2cJjDC+kzb+DhTneuaKfkZCF8gRjafYnyoSyyx4seUBWS2cPljqxFk1OLKK/b/058S9UiSi/TS0bXmmAtPG+TJKpGYb7pVk="""

    def testImportKey7(self):
        for ssh in (self.ssh_pub, tostr(self.ssh_pub)):
            key_obj = DSA.importKey(ssh)
            self.assertFalse(key_obj.has_private())
            self.assertEqual(self.y, key_obj.y)
            self.assertEqual(self.p, key_obj.p)
            self.assertEqual(self.q, key_obj.q)
            self.assertEqual(self.g, key_obj.g)

    def testExportKey7(self):
        tup = (self.y, self.g, self.p, self.q)
        key = DSA.construct(tup)
        encoded = key.export_key('OpenSSH')
        self.assertEqual(self.ssh_pub, encoded)

    # 8. Encrypted OpenSSL/OpenSSH
    pem_private_encrypted="""\
-----BEGIN DSA PRIVATE KEY-----
Proc-Type: 4,ENCRYPTED
DEK-Info: AES-128-CBC,70B6908939D65E9F2EB999E8729788CE

4V6GHRDpCrdZ8MBjbyp5AlGUrjvr2Pn2e2zVxy5RBt4FBj9/pa0ae0nnyUPMLSUU
kKyOR0topRYTVRLElm4qVrb5uNZ3hRwfbklr+pSrB7O9eHz9V5sfOQxyODS07JxK
k1OdOs70/ouMXLF9EWfAZOmWUccZKHNblUwg1p1UrZIz5jXw4dUE/zqhvXh6d+iC
ADsICaBCjCrRQJKDp50h3+ndQjkYBKVH+pj8TiQ79U7lAvdp3+iMghQN6YXs9mdI
gFpWw/f97oWM4GHZFqHJ+VSMNFjBiFhAvYV587d7Lk4dhD8sCfbxj42PnfRgUItc
nnPqHxmhMQozBWzYM4mQuo3XbF2WlsNFbOzFVyGhw1Bx1s91qvXBVWJh2ozrW0s6
HYDV7ZkcTml/4kjA/d+mve6LZ8kuuR1qCiZx6rkffhh1gDN/1Xz3HVvIy/dQ+h9s
5zp7PwUoWbhqp3WCOr156P6gR8qo7OlT6wMh33FSXK/mxikHK136fV2shwTKQVII
rJBvXpj8nACUmi7scKuTWGeUoXa+dwTZVVe+b+L2U1ZM7+h/neTJiXn7u99PFUwu
xVJtxaV37m3aXxtCsPnbBg==
-----END DSA PRIVATE KEY-----"""

    def testImportKey8(self):
        for pem in (self.pem_private_encrypted, tostr(self.pem_private_encrypted)):
            key_obj = DSA.importKey(pem, "PWDTEST")
            self.assertTrue(key_obj.has_private())
            self.assertEqual(self.y, key_obj.y)
            self.assertEqual(self.p, key_obj.p)
            self.assertEqual(self.q, key_obj.q)
            self.assertEqual(self.g, key_obj.g)
            self.assertEqual(self.x, key_obj.x)

    def testExportKey8(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        encoded = key.export_key('PEM', pkcs8=False, passphrase="PWDTEST")
        key = DSA.importKey(encoded, "PWDTEST")
        self.assertEqual(self.y, key.y)
        self.assertEqual(self.p, key.p)
        self.assertEqual(self.q, key.q)
        self.assertEqual(self.g, key.g)
        self.assertEqual(self.x, key.x)

    # 9. Encrypted PKCS8
    # pbeWithMD5AndDES-CBC
    pem_pkcs8_encrypted="""\
-----BEGIN ENCRYPTED PRIVATE KEY-----
MIIBcTAbBgkqhkiG9w0BBQMwDgQI0GC3BJ/jSw8CAggABIIBUHc1cXZpExIE9tC7
7ryiW+5ihtF2Ekurq3e408GYSAu5smJjN2bvQXmzRFBz8W38K8eMf1sbWroZ4+zn
kZSbb9nSm5kAa8lR2+oF2k+WRswMR/PTC3f/D9STO2X0QxdrzKgIHEcSGSHp5jTx
aVvbkCDHo9vhBTl6S3ogZ48As/MEro76+9igUwJ1jNhIQZPJ7e20QH5qDpQFFJN4
CKl2ENSEuwGiqBszItFy4dqH0g63ZGZV/xt9wSO9Rd7SK/EbA/dklOxBa5Y/VItM
gnIhs9XDMoGYyn6F023EicNJm6g/bVQk81BTTma4tm+12TKGdYm+QkeZvCOMZylr
Wv67cKwO3cAXt5C3QXMDgYR64XvuaT5h7C0igMp2afSXJlnbHEbFxQVJlv83T4FM
eZ4k+NQDbEL8GiHmFxzDWQAuPPZKJWEEEV2p/To+WOh+kSDHQw==
-----END ENCRYPTED PRIVATE KEY-----"""

    def testImportKey9(self):
        for pem in (self.pem_pkcs8_encrypted, tostr(self.pem_pkcs8_encrypted)):
            key_obj = DSA.importKey(pem, "PWDTEST")
            self.assertTrue(key_obj.has_private())
            self.assertEqual(self.y, key_obj.y)
            self.assertEqual(self.p, key_obj.p)
            self.assertEqual(self.q, key_obj.q)
            self.assertEqual(self.g, key_obj.g)
            self.assertEqual(self.x, key_obj.x)

    # 10. Encrypted PKCS8
    # pkcs5PBES2 /
    # pkcs5PBKDF2 (rounds=1000, salt=D725BF1B6B8239F4) /
    # des-EDE3-CBC (iv=27A1C66C42AFEECE)
    #
    der_pkcs8_encrypted=\
    '30820196304006092a864886f70d01050d3033301b06092a864886f70d01050c'+\
    '300e0408d725bf1b6b8239f4020203e8301406082a864886f70d0307040827a1'+\
    'c66c42afeece048201505cacfde7bf8edabb3e0d387950dc872662ea7e9b1ed4'+\
    '400d2e7e6186284b64668d8d0328c33a9d9397e6f03df7cb68268b0a06b4e22f'+\
    '7d132821449ecf998a8b696dbc6dd2b19e66d7eb2edfeb4153c1771d49702395'+\
    '4f36072868b5fcccf93413a5ac4b2eb47d4b3f681c6bd67ae363ed776f45ae47'+\
    '174a00098a7c930a50f820b227ddf50f9742d8e950d02586ff2dac0e3c372248'+\
    'e5f9b6a7a02f4004f20c87913e0f7b52bccc209b95d478256a890b31d4c9adec'+\
    '21a4d157a179a93a3dad06f94f3ce486b46dfa7fc15fd852dd7680bbb2f17478'+\
    '7e71bd8dbaf81eca7518d76c1d26256e95424864ba45ca5d47d7c5a421be02fa'+\
    'b94ab01e18593f66cf9094eb5c94b9ecf3aa08b854a195cf87612fbe5e96c426'+\
    '2b0d573e52dc71ba3f5e468c601e816c49b7d32c698b22175e89aaef0c443770'+\
    '5ef2f88a116d99d8e2869a4fd09a771b84b49e4ccb79aadcb1c9'

    def testImportKey10(self):
        key_obj = DSA.importKey(self.der_pkcs8_encrypted, "PWDTEST")
        self.assertTrue(key_obj.has_private())
        self.assertEqual(self.y, key_obj.y)
        self.assertEqual(self.p, key_obj.p)
        self.assertEqual(self.q, key_obj.q)
        self.assertEqual(self.g, key_obj.g)
        self.assertEqual(self.x, key_obj.x)

    def testExportKey10(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        randfunc = BytesIO(unhexlify(b("27A1C66C42AFEECE") + b("D725BF1B6B8239F4"))).read
        encoded = key.export_key('DER', pkcs8=True, passphrase="PWDTEST", randfunc=randfunc)
        self.assertEqual(self.der_pkcs8_encrypted, encoded)

    # ----

    def testImportError1(self):
        self.assertRaises(ValueError, DSA.importKey, self.der_pkcs8_encrypted, "wrongpwd")

    def testExportError2(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        self.assertRaises(ValueError, key.export_key, 'DER', pkcs8=False, passphrase="PWDTEST")

    def test_import_key(self):
        """Verify importKey is an alias to import_key"""

        key_obj = DSA.import_key(self.der_public)
        self.assertFalse(key_obj.has_private())
        self.assertEqual(self.y, key_obj.y)
        self.assertEqual(self.p, key_obj.p)
        self.assertEqual(self.q, key_obj.q)
        self.assertEqual(self.g, key_obj.g)

    def test_exportKey(self):
        tup = (self.y, self.g, self.p, self.q, self.x)
        key = DSA.construct(tup)
        self.assertEqual(key.exportKey(), key.export_key())


    def test_import_empty(self):
        self.assertRaises(ValueError, DSA.import_key, b'')


class ImportKeyFromX509Cert(unittest.TestCase):

    def test_x509v1(self):

        # Sample V1 certificate with a 1024 bit DSA key
        x509_v1_cert = """
-----BEGIN CERTIFICATE-----
MIIDUjCCArsCAQIwDQYJKoZIhvcNAQEFBQAwfjENMAsGA1UEChMEQWNtZTELMAkG
A1UECxMCUkQxHDAaBgkqhkiG9w0BCQEWDXNwYW1AYWNtZS5vcmcxEzARBgNVBAcT
Ck1ldHJvcG9saXMxETAPBgNVBAgTCE5ldyBZb3JrMQswCQYDVQQGEwJVUzENMAsG
A1UEAxMEdGVzdDAeFw0xNDA3MTEyMDM4NDNaFw0xNzA0MDYyMDM4NDNaME0xCzAJ
BgNVBAYTAlVTMREwDwYDVQQIEwhOZXcgWW9yazENMAsGA1UEChMEQWNtZTELMAkG
A1UECxMCUkQxDzANBgNVBAMTBnBvbGFuZDCCAbYwggErBgcqhkjOOAQBMIIBHgKB
gQDOrN4Ox4+t3T6wKeHfhzArhcrNEFMQ4Ss+4PIKyimDy9Bn64WPkL1B/9dvYIga
23GLu6tVJmXo6EdJnVOHEMhr99EeOwuDWWeP7Awq7RSlKEejokr4BEzMTW/tExSD
cO6/GI7xzh0eTH+VTTPDfyrJMYCkh0rJAfCP+5xrmPNetwIVALtXYOV1yoRrzJ2Q
M5uEjidH6GiZAoGAfUqA1SAm5g5U68SILMVX9l5rq0OpB0waBMpJQ31/R/yXNDqo
c3gGWZTOJFU4IzwNpGhrGNADUByz/lc1SAOAdEJIr0JVrhbGewQjB4pWqoLGbBKz
RoavTNDc/zD7SYa12evWDHADwvlXoeQg+lWop1zS8OqaDC7aLGKpWN3/m8kDgYQA
AoGAKoirPAfcp1rbbl4y2FFAIktfW8f4+T7d2iKSg73aiVfujhNOt1Zz1lfC0NI2
eonLWO3tAM4XGKf1TLjb5UXngGn40okPsaA81YE6ZIKm20ywjlOY3QkAEdMaLVY3
9PJvM8RGB9m7pLKxyHfGMfF40MVN4222zKeGp7xhM0CNiCUwDQYJKoZIhvcNAQEF
BQADgYEAfbNZfpYa2KlALEM1FZnwvQDvJHntHz8LdeJ4WM7CXDlKi67wY2HKM30w
s2xej75imkVOFd1kF2d0A8sjfriXLVIt1Hwq9ANZomhu4Edx0xpH8tqdh/bDtnM2
TmduZNY9OWkb07h0CtWD6Zt8fhRllVsSSrlWd/2or7FXNC5weFQ=
-----END CERTIFICATE-----
        """.strip()

        # DSA public key as dumped by openssl
        y_str = """
2a:88:ab:3c:07:dc:a7:5a:db:6e:5e:32:d8:51:40:
22:4b:5f:5b:c7:f8:f9:3e:dd:da:22:92:83:bd:da:
89:57:ee:8e:13:4e:b7:56:73:d6:57:c2:d0:d2:36:
7a:89:cb:58:ed:ed:00:ce:17:18:a7:f5:4c:b8:db:
e5:45:e7:80:69:f8:d2:89:0f:b1:a0:3c:d5:81:3a:
64:82:a6:db:4c:b0:8e:53:98:dd:09:00:11:d3:1a:
2d:56:37:f4:f2:6f:33:c4:46:07:d9:bb:a4:b2:b1:
c8:77:c6:31:f1:78:d0:c5:4d:e3:6d:b6:cc:a7:86:
a7:bc:61:33:40:8d:88:25
        """
        p_str = """
00:ce:ac:de:0e:c7:8f:ad:dd:3e:b0:29:e1:df:87:
30:2b:85:ca:cd:10:53:10:e1:2b:3e:e0:f2:0a:ca:
29:83:cb:d0:67:eb:85:8f:90:bd:41:ff:d7:6f:60:
88:1a:db:71:8b:bb:ab:55:26:65:e8:e8:47:49:9d:
53:87:10:c8:6b:f7:d1:1e:3b:0b:83:59:67:8f:ec:
0c:2a:ed:14:a5:28:47:a3:a2:4a:f8:04:4c:cc:4d:
6f:ed:13:14:83:70:ee:bf:18:8e:f1:ce:1d:1e:4c:
7f:95:4d:33:c3:7f:2a:c9:31:80:a4:87:4a:c9:01:
f0:8f:fb:9c:6b:98:f3:5e:b7
        """
        q_str = """
00:bb:57:60:e5:75:ca:84:6b:cc:9d:90:33:9b:84:
8e:27:47:e8:68:99
        """
        g_str = """
7d:4a:80:d5:20:26:e6:0e:54:eb:c4:88:2c:c5:57:
f6:5e:6b:ab:43:a9:07:4c:1a:04:ca:49:43:7d:7f:
47:fc:97:34:3a:a8:73:78:06:59:94:ce:24:55:38:
23:3c:0d:a4:68:6b:18:d0:03:50:1c:b3:fe:57:35:
48:03:80:74:42:48:af:42:55:ae:16:c6:7b:04:23:
07:8a:56:aa:82:c6:6c:12:b3:46:86:af:4c:d0:dc:
ff:30:fb:49:86:b5:d9:eb:d6:0c:70:03:c2:f9:57:
a1:e4:20:fa:55:a8:a7:5c:d2:f0:ea:9a:0c:2e:da:
2c:62:a9:58:dd:ff:9b:c9
        """

        key = DSA.importKey(x509_v1_cert)
        for comp_name in ('y', 'p', 'q', 'g'):
            comp_str = locals()[comp_name + "_str"]
            comp = int(re.sub("[^0-9a-f]", "", comp_str), 16)
            self.assertEqual(getattr(key, comp_name), comp)
        self.assertFalse(key.has_private())

    def test_x509v3(self):

        # Sample V3 certificate with a 1024 bit DSA key
        x509_v3_cert = """
-----BEGIN CERTIFICATE-----
MIIFhjCCA26gAwIBAgIBAzANBgkqhkiG9w0BAQsFADBhMQswCQYDVQQGEwJVUzEL
MAkGA1UECAwCTUQxEjAQBgNVBAcMCUJhbHRpbW9yZTEQMA4GA1UEAwwHVGVzdCBD
QTEfMB0GCSqGSIb3DQEJARYQdGVzdEBleGFtcGxlLmNvbTAeFw0xNDA3MTMyMDUz
MjBaFw0xNzA0MDgyMDUzMjBaMEAxCzAJBgNVBAYTAlVTMQswCQYDVQQIDAJNRDES
MBAGA1UEBwwJQmFsdGltb3JlMRAwDgYDVQQDDAdhdXN0cmlhMIIBtjCCASsGByqG
SM44BAEwggEeAoGBALfd8gyEpVPA0ZI69Kp3nyJcu5N0ZZ3K1K9hleQLNqKEcZOh
7a/C2J1TPdmHTLJ0rAwBZ1nWxnARSgRphziGDFspKCYQwYcSMz8KoFgvXbXpuchy
oFACiQ2LqZnc5MakuLQtLcQciSYGYj3zmZdYMoa904F1aDWr+DxQI6DVC3/bAhUA
hqXMCJ6fQK3G2O9S3/CC/yVZXCsCgYBRXROl3R2khX7l10LQjDEgo3B1IzjXU/jP
McMBl6XO+nBJXxr/scbq8Ajiv7LTnGpSjgryHtvfj887kfvo8QbSS3kp3vq5uSqI
ui7E7r3jguWaLj616AG1HWOctXJUjqsiabZwsp2h09gHTzmHEXBOmiARu8xFxKAH
xsuo7onAbwOBhAACgYBylWjWSnKHE8mHx1A5m/0GQx6xnhWIe3+MJAnEhRGxA2J4
SCsfWU0OwglIQToh1z5uUU9oDi9cYgNPBevOFRnDhc2yaJY6VAYnI+D+6J5IU6Yd
0iaG/iSc4sV4bFr0axcPpse3SN0XaQxiKeSFBfFnoMqL+dd9Gb3QPZSllBcVD6OB
1TCB0jAdBgNVHQ4EFgQUx5wN0Puotv388M9Tp/fsPbZpzAUwHwYDVR0jBBgwFoAU
a0hkif3RMaraiWtsOOZZlLu9wJwwCQYDVR0TBAIwADALBgNVHQ8EBAMCBeAwSgYD
VR0RBEMwQYILZXhhbXBsZS5jb22CD3d3dy5leGFtcGxlLmNvbYIQbWFpbC5leGFt
cGxlLmNvbYIPZnRwLmV4YW1wbGUuY29tMCwGCWCGSAGG+EIBDQQfFh1PcGVuU1NM
IEdlbmVyYXRlZCBDZXJ0aWZpY2F0ZTANBgkqhkiG9w0BAQsFAAOCAgEAyWf1TiJI
aNEIA9o/PG8/JiGASTS2/HBVTJbkq03k6NkJVk/GxC1DPziTUJ+CdWlHWcAi1EOW
Ach3QxNDRrVfCOfCMDgElIO1094/reJgdFYG00LRi8QkRJuxANV7YS4tLudhyHJC
kR2lhdMNmEuzWK+s2y+5cLrdm7qdvdENQCcV67uvGPx4sc+EaE7x13SczKjWBtbo
QCs6JTOW+EkPRl4Zo27K4OIZ43/J+GxvwU9QUVH3wPVdbbLNw+QeTFBYMTEcxyc4
kv50HPBFaithziXBFyvdIs19FjkFzu0Uz/e0zb1+vMzQlJMD94HVOrMnIj5Sb2cL
KKdYXS4uhxFJmdV091Xur5JkYYwEzuaGav7J3zOzYutrIGTgDluLCvA+VQkRcTsy
jZ065SkY/v+38QHp+cmm8WRluupJTs8wYzVp6Fu0iFaaK7ztFmaZmHpiPIfDFjva
aCIgzzT5NweJd/b71A2SyzHXJ14zBXsr1PMylMp2TpHIidhuuNuQL6I0HaollB4M
Z3FsVBMhVDw4Z76qnFPr8mZE2tar33hSlJI/3pS/bBiukuBk8U7VB0X8OqaUnP3C
7b2Z4G8GtqDVcKGMzkvMjT4n9rKd/Le+qHSsQOGO9W/0LB7UDAZSwUsfAPnoBgdS
5t9tIomLCOstByXi+gGZue1TcdCa3Ph4kO0=
-----END CERTIFICATE-----
        """.strip()

        # DSA public key as dumped by openssl
        y_str = """
72:95:68:d6:4a:72:87:13:c9:87:c7:50:39:9b:fd:
06:43:1e:b1:9e:15:88:7b:7f:8c:24:09:c4:85:11:
b1:03:62:78:48:2b:1f:59:4d:0e:c2:09:48:41:3a:
21:d7:3e:6e:51:4f:68:0e:2f:5c:62:03:4f:05:eb:
ce:15:19:c3:85:cd:b2:68:96:3a:54:06:27:23:e0:
fe:e8:9e:48:53:a6:1d:d2:26:86:fe:24:9c:e2:c5:
78:6c:5a:f4:6b:17:0f:a6:c7:b7:48:dd:17:69:0c:
62:29:e4:85:05:f1:67:a0:ca:8b:f9:d7:7d:19:bd:
d0:3d:94:a5:94:17:15:0f
        """
        p_str = """
00:b7:dd:f2:0c:84:a5:53:c0:d1:92:3a:f4:aa:77:
9f:22:5c:bb:93:74:65:9d:ca:d4:af:61:95:e4:0b:
36:a2:84:71:93:a1:ed:af:c2:d8:9d:53:3d:d9:87:
4c:b2:74:ac:0c:01:67:59:d6:c6:70:11:4a:04:69:
87:38:86:0c:5b:29:28:26:10:c1:87:12:33:3f:0a:
a0:58:2f:5d:b5:e9:b9:c8:72:a0:50:02:89:0d:8b:
a9:99:dc:e4:c6:a4:b8:b4:2d:2d:c4:1c:89:26:06:
62:3d:f3:99:97:58:32:86:bd:d3:81:75:68:35:ab:
f8:3c:50:23:a0:d5:0b:7f:db
        """
        q_str = """
00:86:a5:cc:08:9e:9f:40:ad:c6:d8:ef:52:df:f0:
82:ff:25:59:5c:2b
        """
        g_str = """
51:5d:13:a5:dd:1d:a4:85:7e:e5:d7:42:d0:8c:31:
20:a3:70:75:23:38:d7:53:f8:cf:31:c3:01:97:a5:
ce:fa:70:49:5f:1a:ff:b1:c6:ea:f0:08:e2:bf:b2:
d3:9c:6a:52:8e:0a:f2:1e:db:df:8f:cf:3b:91:fb:
e8:f1:06:d2:4b:79:29:de:fa:b9:b9:2a:88:ba:2e:
c4:ee:bd:e3:82:e5:9a:2e:3e:b5:e8:01:b5:1d:63:
9c:b5:72:54:8e:ab:22:69:b6:70:b2:9d:a1:d3:d8:
07:4f:39:87:11:70:4e:9a:20:11:bb:cc:45:c4:a0:
07:c6:cb:a8:ee:89:c0:6f
        """

        key = DSA.importKey(x509_v3_cert)
        for comp_name in ('y', 'p', 'q', 'g'):
            comp_str = locals()[comp_name + "_str"]
            comp = int(re.sub("[^0-9a-f]", "", comp_str), 16)
            self.assertEqual(getattr(key, comp_name), comp)
        self.assertFalse(key.has_private())


if __name__ == '__main__':
    unittest.main()

def get_tests(config={}):
    tests = []
    tests += list_test_cases(ImportKeyTests)
    tests += list_test_cases(ImportKeyFromX509Cert)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

