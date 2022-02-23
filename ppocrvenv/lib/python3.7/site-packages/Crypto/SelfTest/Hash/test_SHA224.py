# -*- coding: utf-8 -*-
#
#  SelfTest/Hash/test_SHA224.py: Self-test for the SHA-224 hash function
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

"""Self-test suite for Crypto.Hash.SHA224"""

# Test vectors from various sources
# This is a list of (expected_result, input[, description]) tuples.
test_data = [

    # RFC 3874: Section 3.1, "Test Vector #1
    ('23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7', 'abc'),

    # RFC 3874: Section 3.2, "Test Vector #2
    ('75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525', 'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'),

    # RFC 3874: Section 3.3, "Test Vector #3
    ('20794655980c91d8bbb4c1ea97618a4bf03f42581948b2ee4ee7ad67', 'a' * 10**6, "'a' * 10**6"),

    # Examples from http://de.wikipedia.org/wiki/Secure_Hash_Algorithm
    ('d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f', ''),

    ('49b08defa65e644cbf8a2dd9270bdededabc741997d1dadd42026d7b',
     'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern'),

    ('58911e7fccf2971a7d07f93162d8bd13568e71aa8fc86fc1fe9043d1',
     'Frank jagt im komplett verwahrlosten Taxi quer durch Bayern'),

]

def get_tests(config={}):
    from Crypto.Hash import SHA224
    from .common import make_hash_tests
    return make_hash_tests(SHA224, "SHA224", test_data,
        digest_size=28,
        oid='2.16.840.1.101.3.4.2.4')

if __name__ == '__main__':
    import unittest
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
