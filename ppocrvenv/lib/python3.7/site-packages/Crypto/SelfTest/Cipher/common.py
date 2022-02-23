# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/common.py: Common code for Crypto.SelfTest.Hash
#
# Written in 2008 by Dwayne C. Litzenberger <dlitz@dlitz.net>
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

"""Self-testing for PyCrypto hash modules"""

import unittest
from binascii import a2b_hex, b2a_hex, hexlify

from Crypto.Util.py3compat import b
from Crypto.Util.strxor import strxor_c

class _NoDefault: pass        # sentinel object
def _extract(d, k, default=_NoDefault):
    """Get an item from a dictionary, and remove it from the dictionary."""
    try:
        retval = d[k]
    except KeyError:
        if default is _NoDefault:
            raise
        return default
    del d[k]
    return retval

# Generic cipher test case
class CipherSelfTest(unittest.TestCase):

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module

        # Extract the parameters
        params = params.copy()
        self.description = _extract(params, 'description')
        self.key = b(_extract(params, 'key'))
        self.plaintext = b(_extract(params, 'plaintext'))
        self.ciphertext = b(_extract(params, 'ciphertext'))
        self.module_name = _extract(params, 'module_name', None)
        self.assoc_data = _extract(params, 'assoc_data', None)
        self.mac = _extract(params, 'mac', None)
        if self.assoc_data:
            self.mac = b(self.mac)

        mode = _extract(params, 'mode', None)
        self.mode_name = str(mode)

        if mode is not None:
            # Block cipher
            self.mode = getattr(self.module, "MODE_" + mode)

            self.iv = _extract(params, 'iv', None)
            if self.iv is None:
                self.iv = _extract(params, 'nonce', None)
            if self.iv is not None:
                self.iv = b(self.iv)

        else:
            # Stream cipher
            self.mode = None
            self.iv = _extract(params, 'iv', None)
            if self.iv is not None:
                self.iv = b(self.iv)

        self.extra_params = params

    def shortDescription(self):
        return self.description

    def _new(self):
        params = self.extra_params.copy()
        key = a2b_hex(self.key)

        old_style = []
        if self.mode is not None:
            old_style = [ self.mode ]
        if self.iv is not None:
            old_style += [ a2b_hex(self.iv) ]

        return self.module.new(key, *old_style, **params)

    def isMode(self, name):
        if not hasattr(self.module, "MODE_"+name):
            return False
        return self.mode == getattr(self.module, "MODE_"+name)

    def runTest(self):
        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)
        assoc_data = []
        if self.assoc_data:
            assoc_data = [ a2b_hex(b(x)) for x in self.assoc_data]

        ct = None
        pt = None

        #
        # Repeat the same encryption or decryption twice and verify
        # that the result is always the same
        #
        for i in range(2):
            cipher = self._new()
            decipher = self._new()

            # Only AEAD modes
            for comp in assoc_data:
                cipher.update(comp)
                decipher.update(comp)

            ctX = b2a_hex(cipher.encrypt(plaintext))
            ptX = b2a_hex(decipher.decrypt(ciphertext))

            if ct:
                self.assertEqual(ct, ctX)
                self.assertEqual(pt, ptX)
            ct, pt = ctX, ptX

        self.assertEqual(self.ciphertext, ct)  # encrypt
        self.assertEqual(self.plaintext, pt)   # decrypt

        if self.mac:
            mac = b2a_hex(cipher.digest())
            self.assertEqual(self.mac, mac)
            decipher.verify(a2b_hex(self.mac))

class CipherStreamingSelfTest(CipherSelfTest):

    def shortDescription(self):
        desc = self.module_name
        if self.mode is not None:
            desc += " in %s mode" % (self.mode_name,)
        return "%s should behave like a stream cipher" % (desc,)

    def runTest(self):
        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)

        # The cipher should work like a stream cipher

        # Test counter mode encryption, 3 bytes at a time
        ct3 = []
        cipher = self._new()
        for i in range(0, len(plaintext), 3):
            ct3.append(cipher.encrypt(plaintext[i:i+3]))
        ct3 = b2a_hex(b("").join(ct3))
        self.assertEqual(self.ciphertext, ct3)  # encryption (3 bytes at a time)

        # Test counter mode decryption, 3 bytes at a time
        pt3 = []
        cipher = self._new()
        for i in range(0, len(ciphertext), 3):
            pt3.append(cipher.encrypt(ciphertext[i:i+3]))
        # PY3K: This is meant to be text, do not change to bytes (data)
        pt3 = b2a_hex(b("").join(pt3))
        self.assertEqual(self.plaintext, pt3)  # decryption (3 bytes at a time)


class RoundtripTest(unittest.TestCase):
    def __init__(self, module, params):
        from Crypto import Random
        unittest.TestCase.__init__(self)
        self.module = module
        self.iv = Random.get_random_bytes(module.block_size)
        self.key = b(params['key'])
        self.plaintext = 100 * b(params['plaintext'])
        self.module_name = params.get('module_name', None)

    def shortDescription(self):
        return """%s .decrypt() output of .encrypt() should not be garbled""" % (self.module_name,)

    def runTest(self):

        ## ECB mode
        mode = self.module.MODE_ECB
        encryption_cipher = self.module.new(a2b_hex(self.key), mode)
        ciphertext = encryption_cipher.encrypt(self.plaintext)
        decryption_cipher = self.module.new(a2b_hex(self.key), mode)
        decrypted_plaintext = decryption_cipher.decrypt(ciphertext)
        self.assertEqual(self.plaintext, decrypted_plaintext)


class IVLengthTest(unittest.TestCase):
    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.key = b(params['key'])

    def shortDescription(self):
        return "Check that all modes except MODE_ECB and MODE_CTR require an IV of the proper length"

    def runTest(self):
        self.assertRaises(TypeError, self.module.new, a2b_hex(self.key),
                self.module.MODE_ECB, b(""))

    def _dummy_counter(self):
        return "\0" * self.module.block_size


class NoDefaultECBTest(unittest.TestCase):
    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.key = b(params['key'])

    def runTest(self):
        self.assertRaises(TypeError, self.module.new, a2b_hex(self.key))


class BlockSizeTest(unittest.TestCase):
    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module
        self.key = a2b_hex(b(params['key']))

    def runTest(self):
        cipher = self.module.new(self.key, self.module.MODE_ECB)
        self.assertEqual(cipher.block_size, self.module.block_size)


class ByteArrayTest(unittest.TestCase):
    """Verify we can use bytearray's for encrypting and decrypting"""

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module

        # Extract the parameters
        params = params.copy()
        self.description = _extract(params, 'description')
        self.key = b(_extract(params, 'key'))
        self.plaintext = b(_extract(params, 'plaintext'))
        self.ciphertext = b(_extract(params, 'ciphertext'))
        self.module_name = _extract(params, 'module_name', None)
        self.assoc_data = _extract(params, 'assoc_data', None)
        self.mac = _extract(params, 'mac', None)
        if self.assoc_data:
            self.mac = b(self.mac)

        mode = _extract(params, 'mode', None)
        self.mode_name = str(mode)

        if mode is not None:
            # Block cipher
            self.mode = getattr(self.module, "MODE_" + mode)

            self.iv = _extract(params, 'iv', None)
            if self.iv is None:
                self.iv = _extract(params, 'nonce', None)
            if self.iv is not None:
                self.iv = b(self.iv)
        else:
            # Stream cipher
            self.mode = None
            self.iv = _extract(params, 'iv', None)
            if self.iv is not None:
                self.iv = b(self.iv)

        self.extra_params = params

    def _new(self):
        params = self.extra_params.copy()
        key = a2b_hex(self.key)

        old_style = []
        if self.mode is not None:
            old_style = [ self.mode ]
        if self.iv is not None:
            old_style += [ a2b_hex(self.iv) ]

        return self.module.new(key, *old_style, **params)

    def runTest(self):

        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)
        assoc_data = []
        if self.assoc_data:
            assoc_data = [ bytearray(a2b_hex(b(x))) for x in self.assoc_data]

        cipher = self._new()
        decipher = self._new()

        # Only AEAD modes
        for comp in assoc_data:
            cipher.update(comp)
            decipher.update(comp)

        ct = b2a_hex(cipher.encrypt(bytearray(plaintext)))
        pt = b2a_hex(decipher.decrypt(bytearray(ciphertext)))

        self.assertEqual(self.ciphertext, ct)  # encrypt
        self.assertEqual(self.plaintext, pt)   # decrypt

        if self.mac:
            mac = b2a_hex(cipher.digest())
            self.assertEqual(self.mac, mac)
            decipher.verify(bytearray(a2b_hex(self.mac)))


class MemoryviewTest(unittest.TestCase):
    """Verify we can use memoryviews for encrypting and decrypting"""

    def __init__(self, module, params):
        unittest.TestCase.__init__(self)
        self.module = module

        # Extract the parameters
        params = params.copy()
        self.description = _extract(params, 'description')
        self.key = b(_extract(params, 'key'))
        self.plaintext = b(_extract(params, 'plaintext'))
        self.ciphertext = b(_extract(params, 'ciphertext'))
        self.module_name = _extract(params, 'module_name', None)
        self.assoc_data = _extract(params, 'assoc_data', None)
        self.mac = _extract(params, 'mac', None)
        if self.assoc_data:
            self.mac = b(self.mac)

        mode = _extract(params, 'mode', None)
        self.mode_name = str(mode)

        if mode is not None:
            # Block cipher
            self.mode = getattr(self.module, "MODE_" + mode)

            self.iv = _extract(params, 'iv', None)
            if self.iv is None:
                self.iv = _extract(params, 'nonce', None)
            if self.iv is not None:
                self.iv = b(self.iv)
        else:
            # Stream cipher
            self.mode = None
            self.iv = _extract(params, 'iv', None)
            if self.iv is not None:
                self.iv = b(self.iv)

        self.extra_params = params

    def _new(self):
        params = self.extra_params.copy()
        key = a2b_hex(self.key)

        old_style = []
        if self.mode is not None:
            old_style = [ self.mode ]
        if self.iv is not None:
            old_style += [ a2b_hex(self.iv) ]

        return self.module.new(key, *old_style, **params)

    def runTest(self):

        plaintext = a2b_hex(self.plaintext)
        ciphertext = a2b_hex(self.ciphertext)
        assoc_data = []
        if self.assoc_data:
            assoc_data = [ memoryview(a2b_hex(b(x))) for x in self.assoc_data]

        cipher = self._new()
        decipher = self._new()

        # Only AEAD modes
        for comp in assoc_data:
            cipher.update(comp)
            decipher.update(comp)

        ct = b2a_hex(cipher.encrypt(memoryview(plaintext)))
        pt = b2a_hex(decipher.decrypt(memoryview(ciphertext)))

        self.assertEqual(self.ciphertext, ct)  # encrypt
        self.assertEqual(self.plaintext, pt)   # decrypt

        if self.mac:
            mac = b2a_hex(cipher.digest())
            self.assertEqual(self.mac, mac)
            decipher.verify(memoryview(a2b_hex(self.mac)))


def make_block_tests(module, module_name, test_data, additional_params=dict()):
    tests = []
    extra_tests_added = False
    for i in range(len(test_data)):
        row = test_data[i]

        # Build the "params" dictionary with
        # - plaintext
        # - ciphertext
        # - key
        # - mode (default is ECB)
        # - (optionally) description
        # - (optionally) any other parameter that this cipher mode requires
        params = {}
        if len(row) == 3:
            (params['plaintext'], params['ciphertext'], params['key']) = row
        elif len(row) == 4:
            (params['plaintext'], params['ciphertext'], params['key'], params['description']) = row
        elif len(row) == 5:
            (params['plaintext'], params['ciphertext'], params['key'], params['description'], extra_params) = row
            params.update(extra_params)
        else:
            raise AssertionError("Unsupported tuple size %d" % (len(row),))

        if not "mode" in params:
            params["mode"] = "ECB"

        # Build the display-name for the test
        p2 = params.copy()
        p_key = _extract(p2, 'key')
        p_plaintext = _extract(p2, 'plaintext')
        p_ciphertext = _extract(p2, 'ciphertext')
        p_mode = _extract(p2, 'mode')
        p_description = _extract(p2, 'description', None)

        if p_description is not None:
            description = p_description
        elif p_mode == 'ECB' and not p2:
            description = "p=%s, k=%s" % (p_plaintext, p_key)
        else:
            description = "p=%s, k=%s, %r" % (p_plaintext, p_key, p2)
        name = "%s #%d: %s" % (module_name, i+1, description)
        params['description'] = name
        params['module_name'] = module_name
        params.update(additional_params)

        # Add extra test(s) to the test suite before the current test
        if not extra_tests_added:
            tests += [
                RoundtripTest(module, params),
                IVLengthTest(module, params),
                NoDefaultECBTest(module, params),
                ByteArrayTest(module, params),
                BlockSizeTest(module, params),
            ]
            extra_tests_added = True

        # Add the current test to the test suite
        tests.append(CipherSelfTest(module, params))

    return tests

def make_stream_tests(module, module_name, test_data):
    tests = []
    extra_tests_added = False
    for i in range(len(test_data)):
        row = test_data[i]

        # Build the "params" dictionary
        params = {}
        if len(row) == 3:
            (params['plaintext'], params['ciphertext'], params['key']) = row
        elif len(row) == 4:
            (params['plaintext'], params['ciphertext'], params['key'], params['description']) = row
        elif len(row) == 5:
            (params['plaintext'], params['ciphertext'], params['key'], params['description'], extra_params) = row
            params.update(extra_params)
        else:
            raise AssertionError("Unsupported tuple size %d" % (len(row),))

        # Build the display-name for the test
        p2 = params.copy()
        p_key = _extract(p2, 'key')
        p_plaintext = _extract(p2, 'plaintext')
        p_ciphertext = _extract(p2, 'ciphertext')
        p_description = _extract(p2, 'description', None)

        if p_description is not None:
            description = p_description
        elif not p2:
            description = "p=%s, k=%s" % (p_plaintext, p_key)
        else:
            description = "p=%s, k=%s, %r" % (p_plaintext, p_key, p2)
        name = "%s #%d: %s" % (module_name, i+1, description)
        params['description'] = name
        params['module_name'] = module_name

        # Add extra test(s) to the test suite before the current test
        if not extra_tests_added:
            tests += [
                ByteArrayTest(module, params),
            ]

            tests.append(MemoryviewTest(module, params))
            extra_tests_added = True

        # Add the test to the test suite
        tests.append(CipherSelfTest(module, params))
        tests.append(CipherStreamingSelfTest(module, params))
    return tests

# vim:set ts=4 sw=4 sts=4 expandtab:
