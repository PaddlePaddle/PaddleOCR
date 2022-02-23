# -*- coding: utf-8 -*-
#
#  SelfTest/PublicKey/test_ElGamal.py: Self-test for the ElGamal primitive
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

"""Self-test suite for Crypto.PublicKey.ElGamal"""

__revision__ = "$Id$"

import unittest
from Crypto.SelfTest.st_common import list_test_cases, a2b_hex, b2a_hex
from Crypto import Random
from Crypto.PublicKey import ElGamal
from Crypto.Util.number import bytes_to_long
from Crypto.Util.py3compat import *

class ElGamalTest(unittest.TestCase):

    #
    # Test vectors
    #
    # There seem to be no real ElGamal test vectors available in the
    # public domain. The following test vectors have been generated
    # with libgcrypt 1.5.0.
    #
    # Encryption
    tve=[
        {
        # 256 bits
        'p'  :'BA4CAEAAED8CBE952AFD2126C63EB3B345D65C2A0A73D2A3AD4138B6D09BD933',
        'g'  :'05',
        'y'  :'60D063600ECED7C7C55146020E7A31C4476E9793BEAED420FEC9E77604CAE4EF',
        'x'  :'1D391BA2EE3C37FE1BA175A69B2C73A11238AD77675932',
        'k'  :'F5893C5BAB4131264066F57AB3D8AD89E391A0B68A68A1',
        'pt' :'48656C6C6F207468657265',
        'ct1':'32BFD5F487966CEA9E9356715788C491EC515E4ED48B58F0F00971E93AAA5EC7',
        'ct2':'7BE8FBFF317C93E82FCEF9BD515284BA506603FEA25D01C0CB874A31F315EE68'
        },

        {
        # 512 bits
        'p'  :'F1B18AE9F7B4E08FDA9A04832F4E919D89462FD31BF12F92791A93519F75076D6CE3942689CDFF2F344CAFF0F82D01864F69F3AECF566C774CBACF728B81A227',
        'g'  :'07',
        'y'  :'688628C676E4F05D630E1BE39D0066178CA7AA83836B645DE5ADD359B4825A12B02EF4252E4E6FA9BEC1DB0BE90F6D7C8629CABB6E531F472B2664868156E20C',
        'x'  :'14E60B1BDFD33436C0DA8A22FDC14A2CCDBBED0627CE68',
        'k'  :'38DBF14E1F319BDA9BAB33EEEADCAF6B2EA5250577ACE7',
        'pt' :'48656C6C6F207468657265',
        'ct1':'290F8530C2CC312EC46178724F196F308AD4C523CEABB001FACB0506BFED676083FE0F27AC688B5C749AB3CB8A80CD6F7094DBA421FB19442F5A413E06A9772B',
        'ct2':'1D69AAAD1DC50493FB1B8E8721D621D683F3BF1321BE21BC4A43E11B40C9D4D9C80DE3AAC2AB60D31782B16B61112E68220889D53C4C3136EE6F6CE61F8A23A0'
        }
    ]

    # Signature
    tvs=[
        {
        # 256 bits
        'p'  :'D2F3C41EA66530838A704A48FFAC9334F4701ECE3A97CEE4C69DD01AE7129DD7',
        'g'  :'05',
        'y'  :'C3F9417DC0DAFEA6A05C1D2333B7A95E63B3F4F28CC962254B3256984D1012E7',
        'x'  :'165E4A39BE44D5A2D8B1332D416BC559616F536BC735BB',
        'k'  :'C7F0C794A7EAD726E25A47FF8928013680E73C51DD3D7D99BFDA8F492585928F',
        'h'  :'48656C6C6F207468657265',
        'sig1':'35CA98133779E2073EF31165AFCDEB764DD54E96ADE851715495F9C635E1E7C2',
        'sig2':'0135B88B1151279FE5D8078D4FC685EE81177EE9802AB123A73925FC1CB059A7',
        },
        {
        # 512 bits
        'p'  :'E24CF3A4B8A6AF749DCA6D714282FE4AABEEE44A53BB6ED15FBE32B5D3C3EF9CC4124A2ECA331F3C1C1B667ACA3766825217E7B5F9856648D95F05330C6A19CF',
        'g'  :'0B',
        'y'  :'2AD3A1049CA5D4ED207B2431C79A8719BB4073D4A94E450EA6CEE8A760EB07ADB67C0D52C275EE85D7B52789061EE45F2F37D9B2AE522A51C28329766BFE68AC',
        'x'  :'16CBB4F46D9ECCF24FF9F7E63CAA3BD8936341555062AB',
        'k'  :'8A3D89A4E429FD2476D7D717251FB79BF900FFE77444E6BB8299DC3F84D0DD57ABAB50732AE158EA52F5B9E7D8813E81FD9F79470AE22F8F1CF9AEC820A78C69',
        'h'  :'48656C6C6F207468657265',
        'sig1':'BE001AABAFFF976EC9016198FBFEA14CBEF96B000CCC0063D3324016F9E91FE80D8F9325812ED24DDB2B4D4CF4430B169880B3CE88313B53255BD4EC0378586F',
        'sig2':'5E266F3F837BA204E3BBB6DBECC0611429D96F8C7CE8F4EFDF9D4CB681C2A954468A357BF4242CEC7418B51DFC081BCD21299EF5B5A0DDEF3A139A1817503DDE',
        }
    ]

    def test_generate_180(self):
        self._test_random_key(180)

    def test_encryption(self):
        for tv in self.tve:
            d = self.convert_tv(tv, True)
            key = ElGamal.construct(d['key'])
            ct = key._encrypt(d['pt'], d['k'])
            self.assertEqual(ct[0], d['ct1'])
            self.assertEqual(ct[1], d['ct2'])

    def test_decryption(self):
        for tv in self.tve:
            d = self.convert_tv(tv, True)
            key = ElGamal.construct(d['key'])
            pt = key._decrypt((d['ct1'], d['ct2']))
            self.assertEqual(pt, d['pt'])

    def test_signing(self):
        for tv in self.tvs:
            d = self.convert_tv(tv, True)
            key = ElGamal.construct(d['key'])
            sig1, sig2 = key._sign(d['h'], d['k'])
            self.assertEqual(sig1, d['sig1'])
            self.assertEqual(sig2, d['sig2'])

    def test_verification(self):
        for tv in self.tvs:
            d = self.convert_tv(tv, True)
            key = ElGamal.construct(d['key'])
            # Positive test
            res = key._verify( d['h'], (d['sig1'],d['sig2']) )
            self.assertTrue(res)
            # Negative test
            res = key._verify( d['h'], (d['sig1']+1,d['sig2']) )
            self.assertFalse(res)

    def test_bad_key3(self):
        tup = tup0 = list(self.convert_tv(self.tvs[0], 1)['key'])[:3]
        tup[0] += 1 # p += 1 (not prime)
        self.assertRaises(ValueError, ElGamal.construct, tup)

        tup = tup0
        tup[1] = 1 # g = 1
        self.assertRaises(ValueError, ElGamal.construct, tup)

        tup = tup0
        tup[2] = tup[0]*2 # y = 2*p
        self.assertRaises(ValueError, ElGamal.construct, tup)

    def test_bad_key4(self):
        tup = tup0 = list(self.convert_tv(self.tvs[0], 1)['key'])
        tup[3] += 1 # x += 1
        self.assertRaises(ValueError, ElGamal.construct, tup)

    def convert_tv(self, tv, as_longs=0):
        """Convert a test vector from textual form (hexadecimal ascii
        to either integers or byte strings."""
        key_comps = 'p','g','y','x'
        tv2 = {}
        for c in tv.keys():
            tv2[c] = a2b_hex(tv[c])
            if as_longs or c in key_comps or c in ('sig1','sig2'):
                tv2[c] = bytes_to_long(tv2[c])
        tv2['key']=[]
        for c in key_comps:
            tv2['key'] += [tv2[c]]
            del tv2[c]
        return tv2

    def _test_random_key(self, bits):
        elgObj = ElGamal.generate(bits, Random.new().read)
        self._check_private_key(elgObj)
        self._exercise_primitive(elgObj)
        pub = elgObj.publickey()
        self._check_public_key(pub)
        self._exercise_public_primitive(elgObj)

    def _check_private_key(self, elgObj):

        # Check capabilities
        self.assertTrue(elgObj.has_private())

        # Sanity check key data
        self.assertTrue(1<elgObj.g<(elgObj.p-1))
        self.assertEqual(pow(elgObj.g, elgObj.p-1, elgObj.p), 1)
        self.assertTrue(1<elgObj.x<(elgObj.p-1))
        self.assertEqual(pow(elgObj.g, elgObj.x, elgObj.p), elgObj.y)

    def _check_public_key(self, elgObj):

        # Check capabilities
        self.assertFalse(elgObj.has_private())

        # Sanity check key data
        self.assertTrue(1<elgObj.g<(elgObj.p-1))
        self.assertEqual(pow(elgObj.g, elgObj.p-1, elgObj.p), 1)

    def _exercise_primitive(self, elgObj):
        # Test encryption/decryption
        plaintext = 127218
        ciphertext = elgObj._encrypt(plaintext, 123456789)
        plaintextP = elgObj._decrypt(ciphertext)
        self.assertEqual(plaintext, plaintextP)

        # Test signature/verification
        signature = elgObj._sign(plaintext, 987654321)
        elgObj._verify(plaintext, signature)

    def _exercise_public_primitive(self, elgObj):
        plaintext = 92987276
        ciphertext = elgObj._encrypt(plaintext, 123456789)

def get_tests(config={}):
    tests = []
    tests += list_test_cases(ElGamalTest)
    return tests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

