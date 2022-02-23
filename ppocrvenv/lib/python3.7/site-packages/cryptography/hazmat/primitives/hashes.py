# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

import abc
import typing

from cryptography import utils
from cryptography.exceptions import (
    AlreadyFinalized,
)


class HashAlgorithm(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def name(self) -> str:
        """
        A string naming this algorithm (e.g. "sha256", "md5").
        """

    @abc.abstractproperty
    def digest_size(self) -> int:
        """
        The size of the resulting digest in bytes.
        """

    @abc.abstractproperty
    def block_size(self) -> typing.Optional[int]:
        """
        The internal block size of the hash function, or None if the hash
        function does not use blocks internally (e.g. SHA3).
        """


class HashContext(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def algorithm(self) -> HashAlgorithm:
        """
        A HashAlgorithm that will be used by this context.
        """

    @abc.abstractmethod
    def update(self, data: bytes) -> None:
        """
        Processes the provided bytes through the hash.
        """

    @abc.abstractmethod
    def finalize(self) -> bytes:
        """
        Finalizes the hash context and returns the hash digest as bytes.
        """

    @abc.abstractmethod
    def copy(self) -> "HashContext":
        """
        Return a HashContext that is a copy of the current context.
        """


class ExtendableOutputFunction(metaclass=abc.ABCMeta):
    """
    An interface for extendable output functions.
    """


class Hash(HashContext):
    def __init__(
        self,
        algorithm: HashAlgorithm,
        backend: typing.Any = None,
        ctx: typing.Optional["HashContext"] = None,
    ):
        if not isinstance(algorithm, HashAlgorithm):
            raise TypeError("Expected instance of hashes.HashAlgorithm.")
        self._algorithm = algorithm

        if ctx is None:
            from cryptography.hazmat.backends.openssl.backend import (
                backend as ossl,
            )

            self._ctx = ossl.create_hash_ctx(self.algorithm)
        else:
            self._ctx = ctx

    @property
    def algorithm(self) -> HashAlgorithm:
        return self._algorithm

    def update(self, data: bytes) -> None:
        if self._ctx is None:
            raise AlreadyFinalized("Context was already finalized.")
        utils._check_byteslike("data", data)
        self._ctx.update(data)

    def copy(self) -> "Hash":
        if self._ctx is None:
            raise AlreadyFinalized("Context was already finalized.")
        return Hash(self.algorithm, ctx=self._ctx.copy())

    def finalize(self) -> bytes:
        if self._ctx is None:
            raise AlreadyFinalized("Context was already finalized.")
        digest = self._ctx.finalize()
        self._ctx = None
        return digest


class SHA1(HashAlgorithm):
    name = "sha1"
    digest_size = 20
    block_size = 64


class SHA512_224(HashAlgorithm):  # noqa: N801
    name = "sha512-224"
    digest_size = 28
    block_size = 128


class SHA512_256(HashAlgorithm):  # noqa: N801
    name = "sha512-256"
    digest_size = 32
    block_size = 128


class SHA224(HashAlgorithm):
    name = "sha224"
    digest_size = 28
    block_size = 64


class SHA256(HashAlgorithm):
    name = "sha256"
    digest_size = 32
    block_size = 64


class SHA384(HashAlgorithm):
    name = "sha384"
    digest_size = 48
    block_size = 128


class SHA512(HashAlgorithm):
    name = "sha512"
    digest_size = 64
    block_size = 128


class SHA3_224(HashAlgorithm):  # noqa: N801
    name = "sha3-224"
    digest_size = 28
    block_size = None


class SHA3_256(HashAlgorithm):  # noqa: N801
    name = "sha3-256"
    digest_size = 32
    block_size = None


class SHA3_384(HashAlgorithm):  # noqa: N801
    name = "sha3-384"
    digest_size = 48
    block_size = None


class SHA3_512(HashAlgorithm):  # noqa: N801
    name = "sha3-512"
    digest_size = 64
    block_size = None


class SHAKE128(HashAlgorithm, ExtendableOutputFunction):
    name = "shake128"
    block_size = None

    def __init__(self, digest_size: int):
        if not isinstance(digest_size, int):
            raise TypeError("digest_size must be an integer")

        if digest_size < 1:
            raise ValueError("digest_size must be a positive integer")

        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size


class SHAKE256(HashAlgorithm, ExtendableOutputFunction):
    name = "shake256"
    block_size = None

    def __init__(self, digest_size: int):
        if not isinstance(digest_size, int):
            raise TypeError("digest_size must be an integer")

        if digest_size < 1:
            raise ValueError("digest_size must be a positive integer")

        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size


class MD5(HashAlgorithm):
    name = "md5"
    digest_size = 16
    block_size = 64


class BLAKE2b(HashAlgorithm):
    name = "blake2b"
    _max_digest_size = 64
    _min_digest_size = 1
    block_size = 128

    def __init__(self, digest_size: int):

        if digest_size != 64:
            raise ValueError("Digest size must be 64")

        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size


class BLAKE2s(HashAlgorithm):
    name = "blake2s"
    block_size = 64
    _max_digest_size = 32
    _min_digest_size = 1

    def __init__(self, digest_size: int):

        if digest_size != 32:
            raise ValueError("Digest size must be 32")

        self._digest_size = digest_size

    @property
    def digest_size(self) -> int:
        return self._digest_size


class SM3(HashAlgorithm):
    name = "sm3"
    digest_size = 32
    block_size = 64
