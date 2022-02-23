#
#  SelfTest/Util/test_asn.py: Self-test for the Crypto.Util.asn1 module
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

"""Self-tests for Crypto.Util.asn1"""

import unittest

from Crypto.Util.py3compat import *
from Crypto.Util.asn1 import (DerObject, DerSetOf, DerInteger,
                             DerBitString,
                             DerObjectId, DerNull, DerOctetString,
                             DerSequence)

class DerObjectTests(unittest.TestCase):

    def testObjInit1(self):
        # Fail with invalid tag format (must be 1 byte)
        self.assertRaises(ValueError, DerObject, b('\x00\x99'))
        # Fail with invalid implicit tag (must be <0x1F)
        self.assertRaises(ValueError, DerObject, 0x1F)

    # ------

    def testObjEncode1(self):
        # No payload
        der = DerObject(b('\x02'))
        self.assertEqual(der.encode(), b('\x02\x00'))
        # Small payload (primitive)
        der.payload = b('\x45')
        self.assertEqual(der.encode(), b('\x02\x01\x45'))
        # Invariant
        self.assertEqual(der.encode(), b('\x02\x01\x45'))
        # Initialize with numerical tag
        der = DerObject(0x04)
        der.payload = b('\x45')
        self.assertEqual(der.encode(), b('\x04\x01\x45'))
        # Initialize with constructed type
        der = DerObject(b('\x10'), constructed=True)
        self.assertEqual(der.encode(), b('\x30\x00'))

    def testObjEncode2(self):
        # Initialize with payload
        der = DerObject(0x03, b('\x12\x12'))
        self.assertEqual(der.encode(), b('\x03\x02\x12\x12'))

    def testObjEncode3(self):
        # Long payload
        der = DerObject(b('\x10'))
        der.payload = b("0")*128
        self.assertEqual(der.encode(), b('\x10\x81\x80' + "0"*128))

    def testObjEncode4(self):
        # Implicit tags (constructed)
        der = DerObject(0x10, implicit=1, constructed=True)
        der.payload = b('ppll')
        self.assertEqual(der.encode(), b('\xa1\x04ppll'))
        # Implicit tags (primitive)
        der = DerObject(0x02, implicit=0x1E, constructed=False)
        der.payload = b('ppll')
        self.assertEqual(der.encode(), b('\x9E\x04ppll'))

    def testObjEncode5(self):
        # Encode type with explicit tag
        der = DerObject(0x10, explicit=5)
        der.payload = b("xxll")
        self.assertEqual(der.encode(), b("\xa5\x06\x10\x04xxll"))

    # -----

    def testObjDecode1(self):
        # Decode short payload
        der = DerObject(0x02)
        der.decode(b('\x02\x02\x01\x02'))
        self.assertEqual(der.payload, b("\x01\x02"))
        self.assertEqual(der._tag_octet, 0x02)

    def testObjDecode2(self):
        # Decode long payload
        der = DerObject(0x02)
        der.decode(b('\x02\x81\x80' + "1"*128))
        self.assertEqual(der.payload, b("1")*128)
        self.assertEqual(der._tag_octet, 0x02)

    def testObjDecode3(self):
        # Decode payload with too much data gives error
        der = DerObject(0x02)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02\xFF'))
        # Decode payload with too little data gives error
        der = DerObject(0x02)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01'))

    def testObjDecode4(self):
        # Decode implicit tag (primitive)
        der = DerObject(0x02, constructed=False, implicit=0xF)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02'))
        der.decode(b('\x8F\x01\x00'))
        self.assertEqual(der.payload, b('\x00'))
        # Decode implicit tag (constructed)
        der = DerObject(0x02, constructed=True, implicit=0xF)
        self.assertRaises(ValueError, der.decode, b('\x02\x02\x01\x02'))
        der.decode(b('\xAF\x01\x00'))
        self.assertEqual(der.payload, b('\x00'))

    def testObjDecode5(self):
        # Decode payload with unexpected tag gives error
        der = DerObject(0x02)
        self.assertRaises(ValueError, der.decode, b('\x03\x02\x01\x02'))

    def testObjDecode6(self):
        # Arbitrary DER object
        der = DerObject()
        der.decode(b('\x65\x01\x88'))
        self.assertEqual(der._tag_octet, 0x65)
        self.assertEqual(der.payload, b('\x88'))

    def testObjDecode7(self):
        # Decode explicit tag
        der = DerObject(0x10, explicit=5)
        der.decode(b("\xa5\x06\x10\x04xxll"))
        self.assertEqual(der._inner_tag_octet, 0x10)
        self.assertEqual(der.payload, b('xxll'))

        # Explicit tag may be 0
        der = DerObject(0x10, explicit=0)
        der.decode(b("\xa0\x06\x10\x04xxll"))
        self.assertEqual(der._inner_tag_octet, 0x10)
        self.assertEqual(der.payload, b('xxll'))

    def testObjDecode8(self):
        # Verify that decode returns the object
        der = DerObject(0x02)
        self.assertEqual(der, der.decode(b('\x02\x02\x01\x02')))

class DerIntegerTests(unittest.TestCase):

    def testInit1(self):
        der = DerInteger(1)
        self.assertEqual(der.encode(), b('\x02\x01\x01'))

    def testEncode1(self):
        # Single-byte integers
        # Value 0
        der = DerInteger(0)
        self.assertEqual(der.encode(), b('\x02\x01\x00'))
        # Value 1
        der = DerInteger(1)
        self.assertEqual(der.encode(), b('\x02\x01\x01'))
        # Value 127
        der = DerInteger(127)
        self.assertEqual(der.encode(), b('\x02\x01\x7F'))

    def testEncode2(self):
        # Multi-byte integers
        # Value 128
        der = DerInteger(128)
        self.assertEqual(der.encode(), b('\x02\x02\x00\x80'))
        # Value 0x180
        der = DerInteger(0x180)
        self.assertEqual(der.encode(), b('\x02\x02\x01\x80'))
        # One very long integer
        der = DerInteger(2**2048)
        self.assertEqual(der.encode(),
        b('\x02\x82\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00'))

    def testEncode3(self):
        # Negative integers
        # Value -1
        der = DerInteger(-1)
        self.assertEqual(der.encode(), b('\x02\x01\xFF'))
        # Value -128
        der = DerInteger(-128)
        self.assertEqual(der.encode(), b('\x02\x01\x80'))
        # Value
        der = DerInteger(-87873)
        self.assertEqual(der.encode(), b('\x02\x03\xFE\xA8\xBF'))

    def testEncode4(self):
        # Explicit encoding
        number = DerInteger(0x34, explicit=3)
        self.assertEqual(number.encode(), b('\xa3\x03\x02\x01\x34'))

    # -----

    def testDecode1(self):
        # Single-byte integer
        der = DerInteger()
        # Value 0
        der.decode(b('\x02\x01\x00'))
        self.assertEqual(der.value, 0)
        # Value 1
        der.decode(b('\x02\x01\x01'))
        self.assertEqual(der.value, 1)
        # Value 127
        der.decode(b('\x02\x01\x7F'))
        self.assertEqual(der.value, 127)

    def testDecode2(self):
        # Multi-byte integer
        der = DerInteger()
        # Value 0x180L
        der.decode(b('\x02\x02\x01\x80'))
        self.assertEqual(der.value,0x180)
        # One very long integer
        der.decode(
        b('\x02\x82\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00'))
        self.assertEqual(der.value,2**2048)

    def testDecode3(self):
        # Negative integer
        der = DerInteger()
        # Value -1
        der.decode(b('\x02\x01\xFF'))
        self.assertEqual(der.value, -1)
        # Value -32768
        der.decode(b('\x02\x02\x80\x00'))
        self.assertEqual(der.value, -32768)

    def testDecode5(self):
        # We still accept BER integer format
        der = DerInteger()
        # Redundant leading zeroes
        der.decode(b('\x02\x02\x00\x01'))
        self.assertEqual(der.value, 1)
        # Redundant leading 0xFF
        der.decode(b('\x02\x02\xFF\xFF'))
        self.assertEqual(der.value, -1)
        # Empty payload
        der.decode(b('\x02\x00'))
        self.assertEqual(der.value, 0)

    def testDecode6(self):
        # Explicit encoding
        number = DerInteger(explicit=3)
        number.decode(b('\xa3\x03\x02\x01\x34'))
        self.assertEqual(number.value, 0x34)

    def testDecode7(self):
        # Verify decode returns the DerInteger
        der = DerInteger()
        self.assertEqual(der, der.decode(b('\x02\x01\x7F')))

    ###

    def testStrict1(self):
        number = DerInteger()

        number.decode(b'\x02\x02\x00\x01')
        number.decode(b'\x02\x02\x00\x7F')
        self.assertRaises(ValueError, number.decode, b'\x02\x02\x00\x01', strict=True)
        self.assertRaises(ValueError, number.decode, b'\x02\x02\x00\x7F', strict=True)

    ###

    def testErrDecode1(self):
        # Wide length field
        der = DerInteger()
        self.assertRaises(ValueError, der.decode, b('\x02\x81\x01\x01'))


class DerSequenceTests(unittest.TestCase):

    def testInit1(self):
        der = DerSequence([1, DerInteger(2), b('0\x00')])
        self.assertEqual(der.encode(), b('0\x08\x02\x01\x01\x02\x01\x020\x00'))

    def testEncode1(self):
        # Empty sequence
        der = DerSequence()
        self.assertEqual(der.encode(), b('0\x00'))
        self.assertFalse(der.hasOnlyInts())
        # One single-byte integer (zero)
        der.append(0)
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x00'))
        self.assertEqual(der.hasInts(),1)
        self.assertEqual(der.hasInts(False),1)
        self.assertTrue(der.hasOnlyInts())
        self.assertTrue(der.hasOnlyInts(False))
        # Invariant
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x00'))

    def testEncode2(self):
        # Indexing
        der = DerSequence()
        der.append(0)
        der[0] = 1
        self.assertEqual(len(der),1)
        self.assertEqual(der[0],1)
        self.assertEqual(der[-1],1)
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x01'))
        #
        der[:] = [1]
        self.assertEqual(len(der),1)
        self.assertEqual(der[0],1)
        self.assertEqual(der.encode(), b('0\x03\x02\x01\x01'))

    def testEncode3(self):
        # One multi-byte integer (non-zero)
        der = DerSequence()
        der.append(0x180)
        self.assertEqual(der.encode(), b('0\x04\x02\x02\x01\x80'))

    def testEncode4(self):
        # One very long integer
        der = DerSequence()
        der.append(2**2048)
        self.assertEqual(der.encode(), b('0\x82\x01\x05')+
        b('\x02\x82\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00'))

    def testEncode5(self):
        der = DerSequence()
        der += 1
        der += b('\x30\x00')
        self.assertEqual(der.encode(), b('\x30\x05\x02\x01\x01\x30\x00'))

    def testEncode6(self):
        # Two positive integers
        der = DerSequence()
        der.append(0x180)
        der.append(0xFF)
        self.assertEqual(der.encode(), b('0\x08\x02\x02\x01\x80\x02\x02\x00\xff'))
        self.assertTrue(der.hasOnlyInts())
        self.assertTrue(der.hasOnlyInts(False))
        # Two mixed integers
        der = DerSequence()
        der.append(2)
        der.append(-2)
        self.assertEqual(der.encode(), b('0\x06\x02\x01\x02\x02\x01\xFE'))
        self.assertEqual(der.hasInts(), 1)
        self.assertEqual(der.hasInts(False), 2)
        self.assertFalse(der.hasOnlyInts())
        self.assertTrue(der.hasOnlyInts(False))
        #
        der.append(0x01)
        der[1:] = [9,8]
        self.assertEqual(len(der),3)
        self.assertEqual(der[1:],[9,8])
        self.assertEqual(der[1:-1],[9])
        self.assertEqual(der.encode(), b('0\x09\x02\x01\x02\x02\x01\x09\x02\x01\x08'))

    def testEncode7(self):
        # One integer and another type (already encoded)
        der = DerSequence()
        der.append(0x180)
        der.append(b('0\x03\x02\x01\x05'))
        self.assertEqual(der.encode(), b('0\x09\x02\x02\x01\x800\x03\x02\x01\x05'))
        self.assertFalse(der.hasOnlyInts())

    def testEncode8(self):
        # One integer and another type (yet to encode)
        der = DerSequence()
        der.append(0x180)
        der.append(DerSequence([5]))
        self.assertEqual(der.encode(), b('0\x09\x02\x02\x01\x800\x03\x02\x01\x05'))
        self.assertFalse(der.hasOnlyInts())

    ####

    def testDecode1(self):
        # Empty sequence
        der = DerSequence()
        der.decode(b('0\x00'))
        self.assertEqual(len(der),0)
        # One single-byte integer (zero)
        der.decode(b('0\x03\x02\x01\x00'))
        self.assertEqual(len(der),1)
        self.assertEqual(der[0],0)
        # Invariant
        der.decode(b('0\x03\x02\x01\x00'))
        self.assertEqual(len(der),1)
        self.assertEqual(der[0],0)

    def testDecode2(self):
        # One single-byte integer (non-zero)
        der = DerSequence()
        der.decode(b('0\x03\x02\x01\x7f'))
        self.assertEqual(len(der),1)
        self.assertEqual(der[0],127)

    def testDecode4(self):
        # One very long integer
        der = DerSequence()
        der.decode(b('0\x82\x01\x05')+
        b('\x02\x82\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')+
        b('\x00\x00\x00\x00\x00\x00\x00\x00\x00'))
        self.assertEqual(len(der),1)
        self.assertEqual(der[0],2**2048)

    def testDecode6(self):
        # Two integers
        der = DerSequence()
        der.decode(b('0\x08\x02\x02\x01\x80\x02\x02\x00\xff'))
        self.assertEqual(len(der),2)
        self.assertEqual(der[0],0x180)
        self.assertEqual(der[1],0xFF)

    def testDecode7(self):
        # One integer and 2 other types
        der = DerSequence()
        der.decode(b('0\x0A\x02\x02\x01\x80\x24\x02\xb6\x63\x12\x00'))
        self.assertEqual(len(der),3)
        self.assertEqual(der[0],0x180)
        self.assertEqual(der[1],b('\x24\x02\xb6\x63'))
        self.assertEqual(der[2],b('\x12\x00'))

    def testDecode8(self):
        # Only 2 other types
        der = DerSequence()
        der.decode(b('0\x06\x24\x02\xb6\x63\x12\x00'))
        self.assertEqual(len(der),2)
        self.assertEqual(der[0],b('\x24\x02\xb6\x63'))
        self.assertEqual(der[1],b('\x12\x00'))
        self.assertEqual(der.hasInts(), 0)
        self.assertEqual(der.hasInts(False), 0)
        self.assertFalse(der.hasOnlyInts())
        self.assertFalse(der.hasOnlyInts(False))

    def testDecode9(self):
        # Verify that decode returns itself
        der = DerSequence()
        self.assertEqual(der, der.decode(b('0\x06\x24\x02\xb6\x63\x12\x00')))

    ###

    def testErrDecode1(self):
        # Not a sequence
        der = DerSequence()
        self.assertRaises(ValueError, der.decode, b(''))
        self.assertRaises(ValueError, der.decode, b('\x00'))
        self.assertRaises(ValueError, der.decode, b('\x30'))

    def testErrDecode2(self):
        der = DerSequence()
        # Too much data
        self.assertRaises(ValueError, der.decode, b('\x30\x00\x00'))

    def testErrDecode3(self):
        # Wrong length format
        der = DerSequence()
        # Missing length in sub-item
        self.assertRaises(ValueError, der.decode, b('\x30\x04\x02\x01\x01\x00'))
        # Valid BER, but invalid DER length
        self.assertRaises(ValueError, der.decode, b('\x30\x81\x03\x02\x01\x01'))
        self.assertRaises(ValueError, der.decode, b('\x30\x04\x02\x81\x01\x01'))

    def test_expected_nr_elements(self):
        der_bin = DerSequence([1, 2, 3]).encode()

        DerSequence().decode(der_bin, nr_elements=3)
        DerSequence().decode(der_bin, nr_elements=(2,3))
        self.assertRaises(ValueError, DerSequence().decode, der_bin, nr_elements=1)
        self.assertRaises(ValueError, DerSequence().decode, der_bin, nr_elements=(4,5))

    def test_expected_only_integers(self):

        der_bin1 = DerSequence([1, 2, 3]).encode()
        der_bin2 = DerSequence([1, 2, DerSequence([3, 4])]).encode()

        DerSequence().decode(der_bin1, only_ints_expected=True)
        DerSequence().decode(der_bin1, only_ints_expected=False)
        DerSequence().decode(der_bin2, only_ints_expected=False)
        self.assertRaises(ValueError, DerSequence().decode, der_bin2, only_ints_expected=True)


class DerOctetStringTests(unittest.TestCase):

    def testInit1(self):
        der = DerOctetString(b('\xFF'))
        self.assertEqual(der.encode(), b('\x04\x01\xFF'))

    def testEncode1(self):
        # Empty sequence
        der = DerOctetString()
        self.assertEqual(der.encode(), b('\x04\x00'))
        # Small payload
        der.payload = b('\x01\x02')
        self.assertEqual(der.encode(), b('\x04\x02\x01\x02'))

    ####

    def testDecode1(self):
        # Empty sequence
        der = DerOctetString()
        der.decode(b('\x04\x00'))
        self.assertEqual(der.payload, b(''))
        # Small payload
        der.decode(b('\x04\x02\x01\x02'))
        self.assertEqual(der.payload, b('\x01\x02'))

    def testDecode2(self):
        # Verify that decode returns the object
        der = DerOctetString()
        self.assertEqual(der, der.decode(b('\x04\x00')))

    def testErrDecode1(self):
        # No leftovers allowed
        der = DerOctetString()
        self.assertRaises(ValueError, der.decode, b('\x04\x01\x01\xff'))

class DerNullTests(unittest.TestCase):

    def testEncode1(self):
        der = DerNull()
        self.assertEqual(der.encode(), b('\x05\x00'))

    ####

    def testDecode1(self):
        # Empty sequence
        der = DerNull()
        self.assertEqual(der, der.decode(b('\x05\x00')))

class DerObjectIdTests(unittest.TestCase):

    def testInit1(self):
        der = DerObjectId("1.1")
        self.assertEqual(der.encode(), b('\x06\x01)'))

    def testEncode1(self):
        der = DerObjectId('1.2.840.113549.1.1.1')
        self.assertEqual(der.encode(), b('\x06\x09\x2A\x86\x48\x86\xF7\x0D\x01\x01\x01'))
        #
        der = DerObjectId()
        der.value = '1.2.840.113549.1.1.1'
        self.assertEqual(der.encode(), b('\x06\x09\x2A\x86\x48\x86\xF7\x0D\x01\x01\x01'))

    ####

    def testDecode1(self):
        # Empty sequence
        der = DerObjectId()
        der.decode(b('\x06\x09\x2A\x86\x48\x86\xF7\x0D\x01\x01\x01'))
        self.assertEqual(der.value, '1.2.840.113549.1.1.1')

    def testDecode2(self):
        # Verify that decode returns the object
        der = DerObjectId()
        self.assertEqual(der,
                der.decode(b('\x06\x09\x2A\x86\x48\x86\xF7\x0D\x01\x01\x01')))

    def testDecode3(self):
        der = DerObjectId()
        der.decode(b('\x06\x09\x2A\x86\x48\x86\xF7\x0D\x01\x00\x01'))
        self.assertEqual(der.value, '1.2.840.113549.1.0.1')


class DerBitStringTests(unittest.TestCase):

    def testInit1(self):
        der = DerBitString(b("\xFF"))
        self.assertEqual(der.encode(), b('\x03\x02\x00\xFF'))

    def testInit2(self):
        der = DerBitString(DerInteger(1))
        self.assertEqual(der.encode(), b('\x03\x04\x00\x02\x01\x01'))

    def testEncode1(self):
        # Empty sequence
        der = DerBitString()
        self.assertEqual(der.encode(), b('\x03\x01\x00'))
        # Small payload
        der = DerBitString(b('\x01\x02'))
        self.assertEqual(der.encode(), b('\x03\x03\x00\x01\x02'))
        # Small payload
        der = DerBitString()
        der.value = b('\x01\x02')
        self.assertEqual(der.encode(), b('\x03\x03\x00\x01\x02'))

    ####

    def testDecode1(self):
        # Empty sequence
        der = DerBitString()
        der.decode(b('\x03\x00'))
        self.assertEqual(der.value, b(''))
        # Small payload
        der.decode(b('\x03\x03\x00\x01\x02'))
        self.assertEqual(der.value, b('\x01\x02'))

    def testDecode2(self):
        # Verify that decode returns the object
        der = DerBitString()
        self.assertEqual(der, der.decode(b('\x03\x00')))


class DerSetOfTests(unittest.TestCase):

    def testInit1(self):
        der = DerSetOf([DerInteger(1), DerInteger(2)])
        self.assertEqual(der.encode(), b('1\x06\x02\x01\x01\x02\x01\x02'))

    def testEncode1(self):
        # Empty set
        der = DerSetOf()
        self.assertEqual(der.encode(), b('1\x00'))
        # One single-byte integer (zero)
        der.add(0)
        self.assertEqual(der.encode(), b('1\x03\x02\x01\x00'))
        # Invariant
        self.assertEqual(der.encode(), b('1\x03\x02\x01\x00'))

    def testEncode2(self):
        # Two integers
        der = DerSetOf()
        der.add(0x180)
        der.add(0xFF)
        self.assertEqual(der.encode(), b('1\x08\x02\x02\x00\xff\x02\x02\x01\x80'))
        # Initialize with integers
        der = DerSetOf([0x180, 0xFF])
        self.assertEqual(der.encode(), b('1\x08\x02\x02\x00\xff\x02\x02\x01\x80'))

    def testEncode3(self):
        # One integer and another type (no matter what it is)
        der = DerSetOf()
        der.add(0x180)
        self.assertRaises(ValueError, der.add, b('\x00\x02\x00\x00'))

    def testEncode4(self):
        # Only non integers
        der = DerSetOf()
        der.add(b('\x01\x00'))
        der.add(b('\x01\x01\x01'))
        self.assertEqual(der.encode(), b('1\x05\x01\x00\x01\x01\x01'))

    ####

    def testDecode1(self):
        # Empty sequence
        der = DerSetOf()
        der.decode(b('1\x00'))
        self.assertEqual(len(der),0)
        # One single-byte integer (zero)
        der.decode(b('1\x03\x02\x01\x00'))
        self.assertEqual(len(der),1)
        self.assertEqual(list(der),[0])

    def testDecode2(self):
        # Two integers
        der = DerSetOf()
        der.decode(b('1\x08\x02\x02\x01\x80\x02\x02\x00\xff'))
        self.assertEqual(len(der),2)
        l = list(der)
        self.assertTrue(0x180 in l)
        self.assertTrue(0xFF in l)

    def testDecode3(self):
        # One integer and 2 other types
        der = DerSetOf()
        #import pdb; pdb.set_trace()
        self.assertRaises(ValueError, der.decode,
            b('0\x0A\x02\x02\x01\x80\x24\x02\xb6\x63\x12\x00'))

    def testDecode4(self):
        # Verify that decode returns the object
        der = DerSetOf()
        self.assertEqual(der,
                der.decode(b('1\x08\x02\x02\x01\x80\x02\x02\x00\xff')))

    ###

    def testErrDecode1(self):
        # No leftovers allowed
        der = DerSetOf()
        self.assertRaises(ValueError, der.decode,
            b('1\x08\x02\x02\x01\x80\x02\x02\x00\xff\xAA'))

def get_tests(config={}):
    from Crypto.SelfTest.st_common import list_test_cases
    listTests = []
    listTests += list_test_cases(DerObjectTests)
    listTests += list_test_cases(DerIntegerTests)
    listTests += list_test_cases(DerSequenceTests)
    listTests += list_test_cases(DerOctetStringTests)
    listTests += list_test_cases(DerNullTests)
    listTests += list_test_cases(DerObjectIdTests)
    listTests += list_test_cases(DerBitStringTests)
    listTests += list_test_cases(DerSetOfTests)
    return listTests

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
