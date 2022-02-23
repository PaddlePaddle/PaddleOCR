# -*- coding: utf-8 -*-
#
#  Random/random.py : Strong alternative for the standard 'random' module
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

__all__ = ['StrongRandom', 'getrandbits', 'randrange', 'randint', 'choice', 'shuffle', 'sample']

from Crypto import Random

from Crypto.Util.py3compat import is_native_int

class StrongRandom(object):
    def __init__(self, rng=None, randfunc=None):
        if randfunc is None and rng is None:
            self._randfunc = None
        elif randfunc is not None and rng is None:
            self._randfunc = randfunc
        elif randfunc is None and rng is not None:
            self._randfunc = rng.read
        else:
            raise ValueError("Cannot specify both 'rng' and 'randfunc'")

    def getrandbits(self, k):
        """Return an integer with k random bits."""

        if self._randfunc is None:
            self._randfunc = Random.new().read
        mask = (1 << k) - 1
        return mask & bytes_to_long(self._randfunc(ceil_div(k, 8)))

    def randrange(self, *args):
        """randrange([start,] stop[, step]):
        Return a randomly-selected element from range(start, stop, step)."""
        if len(args) == 3:
            (start, stop, step) = args
        elif len(args) == 2:
            (start, stop) = args
            step = 1
        elif len(args) == 1:
            (stop,) = args
            start = 0
            step = 1
        else:
            raise TypeError("randrange expected at most 3 arguments, got %d" % (len(args),))
        if (not is_native_int(start) or not is_native_int(stop) or not
                is_native_int(step)):
            raise TypeError("randrange requires integer arguments")
        if step == 0:
            raise ValueError("randrange step argument must not be zero")

        num_choices = ceil_div(stop - start, step)
        if num_choices < 0:
            num_choices = 0
        if num_choices < 1:
            raise ValueError("empty range for randrange(%r, %r, %r)" % (start, stop, step))

        # Pick a random number in the range of possible numbers
        r = num_choices
        while r >= num_choices:
            r = self.getrandbits(size(num_choices))

        return start + (step * r)

    def randint(self, a, b):
        """Return a random integer N such that a <= N <= b."""
        if not is_native_int(a) or not is_native_int(b):
            raise TypeError("randint requires integer arguments")
        N = self.randrange(a, b+1)
        assert a <= N <= b
        return N

    def choice(self, seq):
        """Return a random element from a (non-empty) sequence.

        If the seqence is empty, raises IndexError.
        """
        if len(seq) == 0:
            raise IndexError("empty sequence")
        return seq[self.randrange(len(seq))]

    def shuffle(self, x):
        """Shuffle the sequence in place."""
        # Fisher-Yates shuffle.  O(n)
        # See http://en.wikipedia.org/wiki/Fisher-Yates_shuffle
        # Working backwards from the end of the array, we choose a random item
        # from the remaining items until all items have been chosen.
        for i in range(len(x)-1, 0, -1):   # iterate from len(x)-1 downto 1
            j = self.randrange(0, i+1)      # choose random j such that 0 <= j <= i
            x[i], x[j] = x[j], x[i]         # exchange x[i] and x[j]

    def sample(self, population, k):
        """Return a k-length list of unique elements chosen from the population sequence."""

        num_choices = len(population)
        if k > num_choices:
            raise ValueError("sample larger than population")

        retval = []
        selected = {}  # we emulate a set using a dict here
        for i in range(k):
            r = None
            while r is None or r in selected:
                r = self.randrange(num_choices)
            retval.append(population[r])
            selected[r] = 1
        return retval

_r = StrongRandom()
getrandbits = _r.getrandbits
randrange = _r.randrange
randint = _r.randint
choice = _r.choice
shuffle = _r.shuffle
sample = _r.sample

# These are at the bottom to avoid problems with recursive imports
from Crypto.Util.number import ceil_div, bytes_to_long, long_to_bytes, size

# vim:set ts=4 sw=4 sts=4 expandtab:
