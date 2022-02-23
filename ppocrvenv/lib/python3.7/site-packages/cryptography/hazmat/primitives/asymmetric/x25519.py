# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import abc

from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives import _serialization


class X25519PublicKey(metaclass=abc.ABCMeta):
    @classmethod
    def from_public_bytes(cls, data: bytes) -> "X25519PublicKey":
        from cryptography.hazmat.backends.openssl.backend import backend

        if not backend.x25519_supported():
            raise UnsupportedAlgorithm(
                "X25519 is not supported by this version of OpenSSL.",
                _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM,
            )

        return backend.x25519_load_public_bytes(data)

    @abc.abstractmethod
    def public_bytes(
        self,
        encoding: _serialization.Encoding,
        format: _serialization.PublicFormat,
    ) -> bytes:
        """
        The serialized bytes of the public key.
        """


class X25519PrivateKey(metaclass=abc.ABCMeta):
    @classmethod
    def generate(cls) -> "X25519PrivateKey":
        from cryptography.hazmat.backends.openssl.backend import backend

        if not backend.x25519_supported():
            raise UnsupportedAlgorithm(
                "X25519 is not supported by this version of OpenSSL.",
                _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM,
            )
        return backend.x25519_generate_key()

    @classmethod
    def from_private_bytes(cls, data: bytes) -> "X25519PrivateKey":
        from cryptography.hazmat.backends.openssl.backend import backend

        if not backend.x25519_supported():
            raise UnsupportedAlgorithm(
                "X25519 is not supported by this version of OpenSSL.",
                _Reasons.UNSUPPORTED_EXCHANGE_ALGORITHM,
            )

        return backend.x25519_load_private_bytes(data)

    @abc.abstractmethod
    def public_key(self) -> X25519PublicKey:
        """
        The serialized bytes of the public key.
        """

    @abc.abstractmethod
    def private_bytes(
        self,
        encoding: _serialization.Encoding,
        format: _serialization.PrivateFormat,
        encryption_algorithm: _serialization.KeySerializationEncryption,
    ) -> bytes:
        """
        The serialized bytes of the private key.
        """

    @abc.abstractmethod
    def exchange(self, peer_public_key: X25519PublicKey) -> bytes:
        """
        Performs a key exchange operation using the provided peer's public key.
        """
