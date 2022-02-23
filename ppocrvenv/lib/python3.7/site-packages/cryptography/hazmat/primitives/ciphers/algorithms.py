# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


from cryptography import utils
from cryptography.hazmat.primitives.ciphers import (
    BlockCipherAlgorithm,
    CipherAlgorithm,
)
from cryptography.hazmat.primitives.ciphers.modes import ModeWithNonce


def _verify_key_size(algorithm: CipherAlgorithm, key: bytes) -> bytes:
    # Verify that the key is instance of bytes
    utils._check_byteslike("key", key)

    # Verify that the key size matches the expected key size
    if len(key) * 8 not in algorithm.key_sizes:
        raise ValueError(
            "Invalid key size ({}) for {}.".format(
                len(key) * 8, algorithm.name
            )
        )
    return key


class AES(CipherAlgorithm, BlockCipherAlgorithm):
    name = "AES"
    block_size = 128
    # 512 added to support AES-256-XTS, which uses 512-bit keys
    key_sizes = frozenset([128, 192, 256, 512])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class Camellia(CipherAlgorithm, BlockCipherAlgorithm):
    name = "camellia"
    block_size = 128
    key_sizes = frozenset([128, 192, 256])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class TripleDES(CipherAlgorithm, BlockCipherAlgorithm):
    name = "3DES"
    block_size = 64
    key_sizes = frozenset([64, 128, 192])

    def __init__(self, key: bytes):
        if len(key) == 8:
            key += key + key
        elif len(key) == 16:
            key += key[:8]
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class Blowfish(CipherAlgorithm, BlockCipherAlgorithm):
    name = "Blowfish"
    block_size = 64
    key_sizes = frozenset(range(32, 449, 8))

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class CAST5(CipherAlgorithm, BlockCipherAlgorithm):
    name = "CAST5"
    block_size = 64
    key_sizes = frozenset(range(40, 129, 8))

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class ARC4(CipherAlgorithm):
    name = "RC4"
    key_sizes = frozenset([40, 56, 64, 80, 128, 160, 192, 256])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class IDEA(CipherAlgorithm):
    name = "IDEA"
    block_size = 64
    key_sizes = frozenset([128])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class SEED(CipherAlgorithm, BlockCipherAlgorithm):
    name = "SEED"
    block_size = 128
    key_sizes = frozenset([128])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class ChaCha20(CipherAlgorithm, ModeWithNonce):
    name = "ChaCha20"
    key_sizes = frozenset([256])

    def __init__(self, key: bytes, nonce: bytes):
        self.key = _verify_key_size(self, key)
        utils._check_byteslike("nonce", nonce)

        if len(nonce) != 16:
            raise ValueError("nonce must be 128-bits (16 bytes)")

        self._nonce = nonce

    @property
    def nonce(self) -> bytes:
        return self._nonce

    @property
    def key_size(self) -> int:
        return len(self.key) * 8


class SM4(CipherAlgorithm, BlockCipherAlgorithm):
    name = "SM4"
    block_size = 128
    key_sizes = frozenset([128])

    def __init__(self, key: bytes):
        self.key = _verify_key_size(self, key)

    @property
    def key_size(self) -> int:
        return len(self.key) * 8
