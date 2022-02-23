# -*- coding: utf-8 -*-
#
#  SelfTest/Cipher/Salsa20.py: Self-test for the Salsa20 stream cipher
#
# Written in 2013 by Fabrizio Tarizzo <fabrizio@fabriziotarizzo.org>
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

"""Self-test suite for Crypto.Cipher.Salsa20"""

import unittest

from Crypto.Util.py3compat import bchr

from Crypto.SelfTest.st_common import list_test_cases

from Crypto.Cipher import Salsa20

from .common import make_stream_tests

# This is a list of (plaintext, ciphertext, key[, description[, params]])
# tuples.
test_data = [
    # Test vectors are taken from
    # http://www.ecrypt.eu.org/stream/svn/viewcvs.cgi/ecrypt/trunk/submissions/salsa20/full/verified.test-vectors
    ( '00' * 512,
      '4dfa5e481da23ea09a31022050859936da52fcee218005164f267cb65f5cfd7f'
    + '2b4f97e0ff16924a52df269515110a07f9e460bc65ef95da58f740b7d1dbb0aa'
    + 'd64cec189c7eb8c6bbf3d7376c80a481d43e628701f6a27afb9fe23919f24114'
    + '8db44f70d7063efcc3dd55a0893a613c3c6fe1c127bd6f59910589293bb6ef9e'
    + 'e24819066dee1a64f49b0bbad5988635272b169af861f85df881939f29ada6fd'
    + '0241410e8d332ae4798d929434a2630de451ec4e0169694cbaa7ebb121ea6a2b'
    + 'da9c1581f429e0a00f7d67e23b730676783b262e8eb43a25f55fb90b3e753aef'
    + '8c6713ec66c51881111593ccb3e8cb8f8de124080501eeeb389c4bcb6977cf95'
    + '7d5789631eb4554400e1e025935dfa7b3e9039d61bdc58a8697d36815bf1985c'
    + 'efdf7ae112e5bb81e37ecf0616ce7147fc08a93a367e08631f23c03b00a8da2f'
    + 'aa5024e5c8d30aca43fc2d5082067b21b234bc741d68fb292c6012c3764ccee3'
    + '1e364a5403e00cfee338a21a01e7d3cefd5a770ca0ab48c435ea6116435f7ad8'
    + '30b217b49f978a68e207ed9f462af7fb195b2115fe8f24f152e4ddc32202d6f2'
    + 'b52fafbcfbc202d8a259a611e901d3f62d065eb13f09bbc45cd45119b843efaa'
    + 'b375703739daced4dd4059fd71c3c47fc2f9939670fad4a46066adcc6a564578'
    + '3308b90ffb72be04a6b147cbe38cc0c3b9267c296a92a7c69873f9f263be9703',
      '80000000000000000000000000000000',
      '128 bits key, set 1, vector 0',
      dict (iv='00'*8)),

    ( '00' * 512,
      'e3be8fdd8beca2e3ea8ef9475b29a6e7003951e1097a5c38d23b7a5fad9f6844'
    + 'b22c97559e2723c7cbbd3fe4fc8d9a0744652a83e72a9c461876af4d7ef1a117'
    + '8da2b74eef1b6283e7e20166abcae538e9716e4669e2816b6b20c5c356802001'
    + 'cc1403a9a117d12a2669f456366d6ebb0f1246f1265150f793cdb4b253e348ae'
    + '203d89bc025e802a7e0e00621d70aa36b7e07cb1e7d5b38d5e222b8b0e4b8407'
    + '0142b1e29504767d76824850320b5368129fdd74e861b498e3be8d16f2d7d169'
    + '57be81f47b17d9ae7c4ff15429a73e10acf250ed3a90a93c711308a74c6216a9'
    + 'ed84cd126da7f28e8abf8bb63517e1ca98e712f4fb2e1a6aed9fdc73291faa17'
    + '958211c4ba2ebd5838c635edb81f513a91a294e194f1c039aeec657dce40aa7e'
    + '7c0af57cacefa40c9f14b71a4b3456a63e162ec7d8d10b8ffb1810d71001b618'
    + '2f9f73da53b85405c11f7b2d890fa8ae0c7f2e926d8a98c7ec4e91b65120e988'
    + '349631a700c6facec3471cb0413656e75e309456584084d7e12c5b43a41c43ed'
    + '9a048abd9b880da65f6a665a20fe7b77cd292fe62cae644b7f7df69f32bdb331'
    + '903e6505ce44fdc293920c6a9ec7057e23df7dad298f82ddf4efb7fdc7bfc622'
    + '696afcfd0cddcc83c7e77f11a649d79acdc3354e9635ff137e929933a0bd6f53'
    + '77efa105a3a4266b7c0d089d08f1e855cc32b15b93784a36e56a76cc64bc8477',
      '8000000000000000000000000000000000000000000000000000000000000000',
      '256 bits key, set 1, vector 0',
      dict (iv='00'*8)),

    ( '00' * 512,
      '169060ccb42bea7bee4d8012a02f3635eb7bca12859fa159cd559094b3507db8'
    + '01735d1a1300102a9c9415546829cbd2021ba217b39b81d89c55b13d0c603359'
    + '3f84159a3c84f4b4f4a0edcd9d38ff261a737909e0b66d68b5cac496f3a5be99'
    + 'cb12c321ab711afaab36cc0947955e1a9bb952ed54425e7711279fbc81bb83f5'
    + '6e55cea44e6daddb05858a153ea6213b3350c12aa1a83ef2726f09485fa71790'
    + 'f9b9f922c7dda1113b1f9d56658ed3402803f511bc1f122601d5e7f0ff036e23'
    + '23ef24bb24195b9fd574823cd8a40c29d86bd35c191e2038779ff696c712b6d8'
    + '2e7014dbe1ac5d527af076c088c4a8d44317958189f6ef54933a7e0816b5b916'
    + 'd8f12ed8afe9422b85e5cc9b8adec9d6cfabe8dbc1082bccc02f5a7266aa074c'
    + 'a284e583a35837798cc0e69d4ce937653b8cdd65ce414b89138615ccb165ad19'
    + '3c6b9c3d05eef4be921a10ea811fe61d11c6867600188e065daff90b509ec56b'
    + 'd41e7e8968c478c78d590c2d2ee24ea009c8f49bc3d81672cfc47895a9e21c9a'
    + '471ebf8e294bee5d2de436ac8d052bf31111b345f1da23c3a4d13b9fc5f0900a'
    + 'a298f98f538973b8fad40d4d159777de2cfe2a3dead1645ddb49794827dba040'
    + 'f70a0ff4ecd155e0f033604693a51e2363880e2ecf98699e7174af7c2c6b0fc6'
    + '59ae329599a3949272a37b9b2183a0910922a3f325ae124dcbdd735364055ceb',
      '09090909090909090909090909090909',
      '128 bits key, set 2, vector 9',
      dict (iv='00'*8)),

    ( '00' * 512,
      '7041e747ceb22ed7812985465f50333124f971da1c5d6efe5ca201b886f31046'
    + 'e757e5c3ec914f60ed1f6bce2819b6810953f12b8ba1199bf82d746a8b8a88f1'
    + '142002978ec4c35b95dc2c82990f9e847a0ab45f2ca72625f5190c820f29f3aa'
    + 'f5f0b5572b06b70a144f2a240c3b3098d4831fa1ce1459f8d1df226a6a79b0ab'
    + '41e91799ef31b5ff3d756c19126b19025858ee70fbd69f2be955cb011c005e31'
    + '32b271b378f39b0cb594e95c99ce6ff17735a541891845bbf0450afcb4a850b9'
    + '4ee90afb713ae7e01295c74381180a3816d7020d5a396c0d97aaa783eaabb6ec'
    + '44d5111157f2212d1b1b8fca7893e8b520cd482418c272ab119b569a2b9598eb'
    + '355624d12e79adab81153b58cd22eaf1b2a32395dedc4a1c66f4d274070b9800'
    + 'ea95766f0245a8295f8aadb36ddbbdfa936417c8dbc6235d19494036964d3e70'
    + 'b125b0f800c3d53881d9d11e7970f827c2f9556935cd29e927b0aceb8cae5fd4'
    + '0fd88a8854010a33db94c96c98735858f1c5df6844f864feaca8f41539313e7f'
    + '3c0610214912cd5e6362197646207e2d64cd5b26c9dfe0822629dcbeb16662e8'
    + '9ff5bf5cf2e499138a5e27bd5027329d0e68ddf53103e9e409523662e27f61f6'
    + '5cf38c1232023e6a6ef66c315bcb2a4328642faabb7ca1e889e039e7c444b34b'
    + 'b3443f596ac730f3df3dfcdb343c307c80f76e43e8898c5e8f43dc3bb280add0',
      '0909090909090909090909090909090909090909090909090909090909090909',
      '256 bits key, set 2, vector 9',
      dict (iv='00'*8)),

    ( '00' * 1024,
      '71daee5142d0728b41b6597933ebf467e43279e30978677078941602629cbf68'
    + 'b73d6bd2c95f118d2b3e6ec955dabb6dc61c4143bc9a9b32b99dbe6866166dc0'
    + '8631b7d6553050303d7252c264d3a90d26c853634813e09ad7545a6ce7e84a5d'
    + 'fc75ec43431207d5319970b0faadb0e1510625bb54372c8515e28e2accf0a993'
    + '0ad15f431874923d2a59e20d9f2a5367dba6051564f150287debb1db536ff9b0'
    + '9ad981f25e5010d85d76ee0c305f755b25e6f09341e0812f95c94f42eead346e'
    + '81f39c58c5faa2c88953dc0cac90469db2063cb5cdb22c9eae22afbf0506fca4'
    + '1dc710b846fbdfe3c46883dd118f3a5e8b11b6afd9e71680d8666557301a2daa'
    + 'fb9496c559784d35a035360885f9b17bd7191977deea932b981ebdb29057ae3c'
    + '92cfeff5e6c5d0cb62f209ce342d4e35c69646ccd14e53350e488bb310a32f8b'
    + '0248e70acc5b473df537ced3f81a014d4083932bedd62ed0e447b6766cd2604b'
    + '706e9b346c4468beb46a34ecf1610ebd38331d52bf33346afec15eefb2a7699e'
    + '8759db5a1f636a48a039688e39de34d995df9f27ed9edc8dd795e39e53d9d925'
    + 'b278010565ff665269042f05096d94da3433d957ec13d2fd82a0066283d0d1ee'
    + 'b81bf0ef133b7fd90248b8ffb499b2414cd4fa003093ff0864575a43749bf596'
    + '02f26c717fa96b1d057697db08ebc3fa664a016a67dcef8807577cc3a09385d3'
    + 'f4dc79b34364bb3b166ce65fe1dd28e3950fe6fa81063f7b16ce1c0e6daac1f8'
    + '188455b77752045e863c9b256ad92bc6e2d08314c5bba191c274f42dfbb3d652'
    + 'bb771956555e880f84cd8b827a4c5a52f3a099fa0259bd4aac3efd541f191170'
    + '4412d6e85fbcc628b335875b9fef24807f6e1bc66c3186159e1e7f5a13913e02'
    + 'd241ce2efdbcaa275039fb14eac5923d17ffbc7f1abd3b45e92127575bfbabf9'
    + '3a257ebef0aa1437b326e41b585af572f7239c33b32981a1577a4f629b027e1e'
    + 'b49d58cc497e944d79cef44357c2bf25442ab779651e991147bf79d6fd3a8868'
    + '0cd3b1748e07fd10d78aceef6db8a5e563570d40127f754146c34a440f2a991a'
    + '23fa39d365141f255041f2135c5cba4373452c114da1801bacca38610e3a6524'
    + '2b822d32de4ab5a7d3cf9b61b37493c863bd12e2cae10530cddcda2cb7a5436b'
    + 'ef8988d4d24e8cdc31b2d2a3586340bc5141f8f6632d0dd543bfed81eb471ba1'
    + 'f3dc2225a15ffddcc03eb48f44e27e2aa390598adf83f15c6608a5f18d4dfcf0'
    + 'f547d467a4d70b281c83a595d7660d0b62de78b9cca023cca89d7b1f83484638'
    + '0e228c25f049184a612ef5bb3d37454e6cfa5b10dceda619d898a699b3c8981a'
    + '173407844bb89b4287bf57dd6600c79e352c681d74b03fa7ea0d7bf6ad69f8a6'
    + '8ecb001963bd2dd8a2baa0083ec09751cd9742402ad716be16d5c052304cfca1',
      '0F62B5085BAE0154A7FA4DA0F34699EC',
      '128 bits key, Set 6, vector#  3',
      dict (iv='288FF65DC42B92F9')),

    ( '00' * 1024,
      '5e5e71f90199340304abb22a37b6625bf883fb89ce3b21f54a10b81066ef87da'
    + '30b77699aa7379da595c77dd59542da208e5954f89e40eb7aa80a84a6176663f'
    + 'd910cde567cf1ff60f7040548d8f376bfd1f44c4774aac37410ede7d5c3463fc'
    + '4508a603201d8495ad257894e5eb1914b53e8da5e4bf2bc83ac87ce55cc67df7'
    + '093d9853d2a83a9c8be969175df7c807a17156df768445dd0874a9271c6537f5'
    + 'ce0466473582375f067fa4fcdaf65dbc0139cd75e8c21a482f28c0fb8c3d9f94'
    + '22606cc8e88fe28fe73ec3cb10ff0e8cc5f2a49e540f007265c65b7130bfdb98'
    + '795b1df9522da46e48b30e55d9f0d787955ece720205b29c85f3ad9be33b4459'
    + '7d21b54d06c9a60b04b8e640c64e566e51566730e86cf128ab14174f91bd8981'
    + 'a6fb00fe587bbd6c38b5a1dfdb04ea7e61536fd229f957aa9b070ca931358e85'
    + '11b92c53c523cb54828fb1513c5636fa9a0645b4a3c922c0db94986d92f314ff'
    + '7852c03b231e4dceea5dd8cced621869cff818daf3c270ff3c8be2e5c74be767'
    + 'a4e1fdf3327a934fe31e46df5a74ae2021cee021d958c4f615263d99a5ddae7f'
    + 'eab45e6eccbafefe4761c57750847b7e75ee2e2f14333c0779ce4678f47b1e1b'
    + '760a03a5f17d6e91d4b42313b3f1077ee270e432fe04917ed1fc8babebf7c941'
    + '42b80dfb44a28a2a3e59093027606f6860bfb8c2e5897078cfccda7314c70035'
    + 'f137de6f05daa035891d5f6f76e1df0fce1112a2ff0ac2bd3534b5d1bf4c7165'
    + 'fb40a1b6eacb7f295711c4907ae457514a7010f3a342b4427593d61ba993bc59'
    + '8bd09c56b9ee53aac5dd861fa4b4bb53888952a4aa9d8ca8671582de716270e1'
    + '97375b3ee49e51fa2bf4ef32015dd9a764d966aa2ae541592d0aa650849e99ca'
    + '5c6c39beebf516457cc32fe4c105bff314a12f1ec94bdf4d626f5d9b1cbbde42'
    + 'e5733f0885765ba29e2e82c829d312f5fc7e180679ac84826c08d0a644b326d0'
    + '44da0fdcc75fa53cfe4ced0437fa4df5a7ecbca8b4cb7c4a9ecf9a60d00a56eb'
    + '81da52adc21f508dbb60a9503a3cc94a896616d86020d5b0e5c637329b6d396a'
    + '41a21ba2c4a9493cf33fa2d4f10f77d5b12fdad7e478ccfe79b74851fc96a7ca'
    + '6320c5efd561a222c0ab0fb44bbda0e42149611d2262bb7d1719150fa798718a'
    + '0eec63ee297cad459869c8b0f06c4e2b56cbac03cd2605b2a924efedf85ec8f1'
    + '9b0b6c90e7cbd933223ffeb1b3a3f9677657905829294c4c70acdb8b0891b47d'
    + '0875d0cd6c0f4efe2917fc44b581ef0d1e4280197065d07da34ab33283364552'
    + 'efad0bd9257b059acdd0a6f246812feb69e7e76065f27dbc2eee94da9cc41835'
    + 'bf826e36e5cebe5d4d6a37a6a666246290ce51a0c082718ab0ec855668db1add'
    + 'a658e5f257e0db39384d02e6145c4c00eaa079098f6d820d872de711b6ed08cf',
      '0F62B5085BAE0154A7FA4DA0F34699EC3F92E5388BDE3184D72A7DD02376C91C',
      '256 bits key, Set 6, vector#  3',
      dict (iv='288FF65DC42B92F9')),

]


class KeyLength(unittest.TestCase):

    def runTest(self):

        nonce = bchr(0) * 8
        for key_length in (15, 30, 33):
            key = bchr(1) * key_length
            self.assertRaises(ValueError, Salsa20.new, key, nonce)


class NonceTests(unittest.TestCase):

    def test_invalid_nonce_length(self):
        key = bchr(1) * 16
        self.assertRaises(ValueError, Salsa20.new, key, bchr(0) * 7)
        self.assertRaises(ValueError, Salsa20.new, key, bchr(0) * 9)

    def test_default_nonce(self):

        cipher1 = Salsa20.new(bchr(1) * 16)
        cipher2 = Salsa20.new(bchr(1) * 16)
        self.assertEqual(len(cipher1.nonce), 8)
        self.assertNotEqual(cipher1.nonce, cipher2.nonce)


class ByteArrayTest(unittest.TestCase):
    """Verify we can encrypt or decrypt bytearrays"""

    def runTest(self):

        data = b"0123"
        key = b"9" * 32
        nonce = b"t" * 8

        # Encryption
        data_ba = bytearray(data)
        key_ba = bytearray(key)
        nonce_ba = bytearray(nonce)

        cipher1 = Salsa20.new(key=key, nonce=nonce)
        ct = cipher1.encrypt(data)

        cipher2 = Salsa20.new(key=key_ba, nonce=nonce_ba)
        key_ba[:1] = b'\xFF'
        nonce_ba[:1] = b'\xFF'
        ct_test = cipher2.encrypt(data_ba)

        self.assertEqual(ct, ct_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decryption
        key_ba = bytearray(key)
        nonce_ba = bytearray(nonce)
        ct_ba = bytearray(ct)

        cipher3 = Salsa20.new(key=key_ba, nonce=nonce_ba)
        key_ba[:1] = b'\xFF'
        nonce_ba[:1] = b'\xFF'
        pt_test = cipher3.decrypt(ct_ba)

        self.assertEqual(data, pt_test)


class MemoryviewTest(unittest.TestCase):
    """Verify we can encrypt or decrypt bytearrays"""

    def runTest(self):

        data = b"0123"
        key = b"9" * 32
        nonce = b"t" * 8

        # Encryption
        data_mv = memoryview(bytearray(data))
        key_mv = memoryview(bytearray(key))
        nonce_mv = memoryview(bytearray(nonce))

        cipher1 = Salsa20.new(key=key, nonce=nonce)
        ct = cipher1.encrypt(data)

        cipher2 = Salsa20.new(key=key_mv, nonce=nonce_mv)
        key_mv[:1] = b'\xFF'
        nonce_mv[:1] = b'\xFF'
        ct_test = cipher2.encrypt(data_mv)

        self.assertEqual(ct, ct_test)
        self.assertEqual(cipher1.nonce, cipher2.nonce)

        # Decryption
        key_mv = memoryview(bytearray(key))
        nonce_mv = memoryview(bytearray(nonce))
        ct_mv = memoryview(bytearray(ct))

        cipher3 = Salsa20.new(key=key_mv, nonce=nonce_mv)
        key_mv[:1] = b'\xFF'
        nonce_mv[:1] = b'\xFF'
        pt_test = cipher3.decrypt(ct_mv)

        self.assertEqual(data, pt_test)


class TestOutput(unittest.TestCase):

    def runTest(self):
        # Encrypt/Decrypt data and test output parameter

        key = b'4' * 32
        nonce = b'5' * 8
        cipher = Salsa20.new(key=key, nonce=nonce)

        pt = b'5' * 300
        ct = cipher.encrypt(pt)

        output = bytearray(len(pt))
        cipher = Salsa20.new(key=key, nonce=nonce)
        res = cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)
        self.assertEqual(res, None)

        cipher = Salsa20.new(key=key, nonce=nonce)
        res = cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)
        self.assertEqual(res, None)

        output = memoryview(bytearray(len(pt)))
        cipher = Salsa20.new(key=key, nonce=nonce)
        cipher.encrypt(pt, output=output)
        self.assertEqual(ct, output)

        cipher = Salsa20.new(key=key, nonce=nonce)
        cipher.decrypt(ct, output=output)
        self.assertEqual(pt, output)

        cipher = Salsa20.new(key=key, nonce=nonce)
        self.assertRaises(TypeError, cipher.encrypt, pt, output=b'0'*len(pt))

        cipher = Salsa20.new(key=key, nonce=nonce)
        self.assertRaises(TypeError, cipher.decrypt, ct, output=b'0'*len(ct))

        shorter_output = bytearray(len(pt) - 1)

        cipher = Salsa20.new(key=key, nonce=nonce)
        self.assertRaises(ValueError, cipher.encrypt, pt, output=shorter_output)

        cipher = Salsa20.new(key=key, nonce=nonce)
        self.assertRaises(ValueError, cipher.decrypt, ct, output=shorter_output)


def get_tests(config={}):
    tests = make_stream_tests(Salsa20, "Salsa20", test_data)
    tests.append(KeyLength())
    tests += list_test_cases(NonceTests)
    tests.append(ByteArrayTest())
    tests.append(MemoryviewTest())
    tests.append(TestOutput())

    return tests


if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
