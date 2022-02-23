# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.


import typing

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives._asymmetric import (
    AsymmetricPadding as AsymmetricPadding,
)
from cryptography.hazmat.primitives.asymmetric import rsa


class PKCS1v15(AsymmetricPadding):
    name = "EMSA-PKCS1-v1_5"


class PSS(AsymmetricPadding):
    MAX_LENGTH = object()
    name = "EMSA-PSS"

    def __init__(self, mgf, salt_length):
        self._mgf = mgf

        if (
            not isinstance(salt_length, int)
            and salt_length is not self.MAX_LENGTH
        ):
            raise TypeError("salt_length must be an integer.")

        if salt_length is not self.MAX_LENGTH and salt_length < 0:
            raise ValueError("salt_length must be zero or greater.")

        self._salt_length = salt_length


class OAEP(AsymmetricPadding):
    name = "EME-OAEP"

    def __init__(
        self,
        mgf: "MGF1",
        algorithm: hashes.HashAlgorithm,
        label: typing.Optional[bytes],
    ):
        if not isinstance(algorithm, hashes.HashAlgorithm):
            raise TypeError("Expected instance of hashes.HashAlgorithm.")

        self._mgf = mgf
        self._algorithm = algorithm
        self._label = label


class MGF1(object):
    MAX_LENGTH = object()

    def __init__(self, algorithm: hashes.HashAlgorithm):
        if not isinstance(algorithm, hashes.HashAlgorithm):
            raise TypeError("Expected instance of hashes.HashAlgorithm.")

        self._algorithm = algorithm


def calculate_max_pss_salt_length(
    key: typing.Union["rsa.RSAPrivateKey", "rsa.RSAPublicKey"],
    hash_algorithm: hashes.HashAlgorithm,
) -> int:
    if not isinstance(key, (rsa.RSAPrivateKey, rsa.RSAPublicKey)):
        raise TypeError("key must be an RSA public or private key")
    # bit length - 1 per RFC 3447
    emlen = (key.key_size + 6) // 8
    salt_length = emlen - hash_algorithm.digest_size - 2
    assert salt_length >= 0
    return salt_length
