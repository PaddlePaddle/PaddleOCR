# -*- coding: utf-8 -*-
#
#  SelfTest/Util/test_generic.py: Self-test for the Crypto.Random.new() function
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

"""Self-test suite for Crypto.Random.new()"""

import sys
import unittest
from Crypto.Util.py3compat import b

class SimpleTest(unittest.TestCase):
    def runTest(self):
        """Crypto.Random.new()"""
        # Import the Random module and try to use it
        from Crypto import Random
        randobj = Random.new()
        x = randobj.read(16)
        y = randobj.read(16)
        self.assertNotEqual(x, y)
        z = Random.get_random_bytes(16)
        self.assertNotEqual(x, z)
        self.assertNotEqual(y, z)
        # Test the Random.random module, which
        # implements a subset of Python's random API
        # Not implemented:
        # seed(), getstate(), setstate(), jumpahead()
        # random(), uniform(), triangular(), betavariate()
        # expovariate(), gammavariate(), gauss(),
        # longnormvariate(), normalvariate(),
        # vonmisesvariate(), paretovariate()
        # weibullvariate()
        # WichmannHill(), whseed(), SystemRandom()
        from Crypto.Random import random
        x = random.getrandbits(16*8)
        y = random.getrandbits(16*8)
        self.assertNotEqual(x, y)
        # Test randrange
        if x>y:
            start = y
            stop = x
        else:
            start = x
            stop = y
        for step in range(1,10):
            x = random.randrange(start,stop,step)
            y = random.randrange(start,stop,step)
            self.assertNotEqual(x, y)
            self.assertEqual(start <= x < stop, True)
            self.assertEqual(start <= y < stop, True)
            self.assertEqual((x - start) % step, 0)
            self.assertEqual((y - start) % step, 0)
        for i in range(10):
            self.assertEqual(random.randrange(1,2), 1)
        self.assertRaises(ValueError, random.randrange, start, start)
        self.assertRaises(ValueError, random.randrange, stop, start, step)
        self.assertRaises(TypeError, random.randrange, start, stop, step, step)
        self.assertRaises(TypeError, random.randrange, start, stop, "1")
        self.assertRaises(TypeError, random.randrange, "1", stop, step)
        self.assertRaises(TypeError, random.randrange, 1, "2", step)
        self.assertRaises(ValueError, random.randrange, start, stop, 0)
        # Test randint
        x = random.randint(start,stop)
        y = random.randint(start,stop)
        self.assertNotEqual(x, y)
        self.assertEqual(start <= x <= stop, True)
        self.assertEqual(start <= y <= stop, True)
        for i in range(10):
            self.assertEqual(random.randint(1,1), 1)
        self.assertRaises(ValueError, random.randint, stop, start)
        self.assertRaises(TypeError, random.randint, start, stop, step)
        self.assertRaises(TypeError, random.randint, "1", stop)
        self.assertRaises(TypeError, random.randint, 1, "2")
        # Test choice
        seq = range(10000)
        x = random.choice(seq)
        y = random.choice(seq)
        self.assertNotEqual(x, y)
        self.assertEqual(x in seq, True)
        self.assertEqual(y in seq, True)
        for i in range(10):
            self.assertEqual(random.choice((1,2,3)) in (1,2,3), True)
        self.assertEqual(random.choice([1,2,3]) in [1,2,3], True)
        if sys.version_info[0] == 3:
            self.assertEqual(random.choice(bytearray(b('123'))) in bytearray(b('123')), True)
        self.assertEqual(1, random.choice([1]))
        self.assertRaises(IndexError, random.choice, [])
        self.assertRaises(TypeError, random.choice, 1)
        # Test shuffle. Lacks random parameter to specify function.
        # Make copies of seq
        seq = range(500)
        x = list(seq)
        y = list(seq)
        random.shuffle(x)
        random.shuffle(y)
        self.assertNotEqual(x, y)
        self.assertEqual(len(seq), len(x))
        self.assertEqual(len(seq), len(y))
        for i in range(len(seq)):
           self.assertEqual(x[i] in seq, True)
           self.assertEqual(y[i] in seq, True)
           self.assertEqual(seq[i] in x, True)
           self.assertEqual(seq[i] in y, True)
        z = [1]
        random.shuffle(z)
        self.assertEqual(z, [1])
        if sys.version_info[0] == 3:
            z = bytearray(b('12'))
            random.shuffle(z)
            self.assertEqual(b('1') in z, True)
            self.assertRaises(TypeError, random.shuffle, b('12'))
        self.assertRaises(TypeError, random.shuffle, 1)
        self.assertRaises(TypeError, random.shuffle, "11")
        self.assertRaises(TypeError, random.shuffle, (1,2))
        # 2to3 wraps a list() around it, alas - but I want to shoot
        # myself in the foot here! :D
        # if sys.version_info[0] == 3:
            # self.assertRaises(TypeError, random.shuffle, range(3))
        # Test sample
        x = random.sample(seq, 20)
        y = random.sample(seq, 20)
        self.assertNotEqual(x, y)
        for i in range(20):
           self.assertEqual(x[i] in seq, True)
           self.assertEqual(y[i] in seq, True)
        z = random.sample([1], 1)
        self.assertEqual(z, [1])
        z = random.sample((1,2,3), 1)
        self.assertEqual(z[0] in (1,2,3), True)
        z = random.sample("123", 1)
        self.assertEqual(z[0] in "123", True)
        z = random.sample(range(3), 1)
        self.assertEqual(z[0] in range(3), True)
        if sys.version_info[0] == 3:
                z = random.sample(b("123"), 1)
                self.assertEqual(z[0] in b("123"), True)
                z = random.sample(bytearray(b("123")), 1)
                self.assertEqual(z[0] in bytearray(b("123")), True)
        self.assertRaises(TypeError, random.sample, 1)

def get_tests(config={}):
    return [SimpleTest()]

if __name__ == '__main__':
    suite = lambda: unittest.TestSuite(get_tests())
    unittest.main(defaultTest='suite')

# vim:set ts=4 sw=4 sts=4 expandtab:
