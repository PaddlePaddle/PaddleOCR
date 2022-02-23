# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/test_SHA512.py: Self-test for the SHA-512 hash function
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

"""Self-test suite for Crypto.Hash.SHA512"""

from binascii import hexlify

from Crypto.Hash import SHA512
from .common import make_hash_tests
from Crypto.SelfTest.loader import load_test_vectors

# Test vectors from various sources
# This is a list of (expected_result, input[, description]) tuples.
test_data_512_other = [

    # RFC 4634: Section Page 8.4, "Test 1"
    ('ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f', 'abc'),

    # RFC 4634: Section Page 8.4, "Test 2.1"
    ('8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909', 'abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu'),

    # RFC 4634: Section Page 8.4, "Test 3"
    ('e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b', 'a' * 10**6, "'a' * 10**6"),

    # Taken from http://de.wikipedia.org/wiki/Secure_Hash_Algorithm
    ('cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e', ''),

    ('af9ed2de700433b803240a552b41b5a472a6ef3fe1431a722b2063c75e9f07451f67a28e37d09cde769424c96aea6f8971389db9e1993d6c565c3c71b855723c', 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'),
]


def get_tests_SHA512():

    test_vectors = load_test_vectors(("Hash", "SHA2"),
                                "SHA512ShortMsg.rsp",
                                "KAT SHA-512",
                                {"len": lambda x: int(x)}) or []

    test_data = test_data_512_other[:]
    for tv in test_vectors:
        try:
            if tv.startswith('['):
                continue
        except AttributeError:
            pass
        if tv.len == 0:
            tv.msg = b""
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))

    tests = make_hash_tests(SHA512, "SHA512", test_data,
                            digest_size=64,
                            oid="2.16.840.1.101.3.4.2.3")
    return tests


def get_tests_SHA512_224():

    test_vectors = load_test_vectors(("Hash", "SHA2"),
                                "SHA512_224ShortMsg.rsp",
                                "KAT SHA-512/224",
                                {"len": lambda x: int(x)}) or []

    test_data = []
    for tv in test_vectors:
        try:
            if tv.startswith('['):
                continue
        except AttributeError:
            pass
        if tv.len == 0:
            tv.msg = b""
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))

    tests = make_hash_tests(SHA512, "SHA512/224", test_data,
                            digest_size=28,
                            oid="2.16.840.1.101.3.4.2.5",
                            extra_params={ "truncate" : "224" })
    return tests


def get_tests_SHA512_256():

    test_vectors = load_test_vectors(("Hash", "SHA2"),
                                "SHA512_256ShortMsg.rsp",
                                "KAT SHA-512/256",
                                {"len": lambda x: int(x)}) or []

    test_data = []
    for tv in test_vectors:
        try:
            if tv.startswith('['):
                continue
        except AttributeError:
            pass
        if tv.len == 0:
            tv.msg = b""
        test_data.append((hexlify(tv.md), tv.msg, tv.desc))

    tests = make_hash_tests(SHA512, "SHA512/256", test_data,
                            digest_size=32,
                            oid="2.16.840.1.101.3.4.2.6",
                            extra_params={ "truncate" : "256" })
    return tests


def get_tests(config={}):

    tests = []
    tests += get_tests_SHA512()
    tests += get_tests_SHA512_224()
    tests += get_tests_SHA512_256()
    return tests

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
