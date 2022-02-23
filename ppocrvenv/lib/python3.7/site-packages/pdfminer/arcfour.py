""" Python implementation of Arcfour encryption algorithm.
See https://en.wikipedia.org/wiki/RC4
This code is in the public domain.

"""


from typing import Sequence


class Arcfour:

    def __init__(self, key: Sequence[int]) -> None:
        # because Py3 range is not indexable
        s = [i for i in range(256)]
        j = 0
        klen = len(key)
        for i in range(256):
            j = (j + s[i] + key[i % klen]) % 256
            (s[i], s[j]) = (s[j], s[i])
        self.s = s
        (self.i, self.j) = (0, 0)
        return

    def process(self, data: bytes) -> bytes:
        (i, j) = (self.i, self.j)
        s = self.s
        r = b''
        for c in iter(data):
            i = (i+1) % 256
            j = (j+s[i]) % 256
            (s[i], s[j]) = (s[j], s[i])
            k = s[(s[i]+s[j]) % 256]
            r += bytes((c ^ k,))
        (self.i, self.j) = (i, j)
        return r

    encrypt = decrypt = process
